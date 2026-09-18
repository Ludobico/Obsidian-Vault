#streaming-pipeline

# 워커 스레드부터 클라이언트까지 — `_run_models()`

> 파일: `chat/process_message.py:1161-1442`(`_run_models`, `_run_model`), `:1498-1605`(`_drain_to_completion`),
> `chat/emitter.py`, `chat/chat_state.py`

[[agent-loop]] (`run_llm_loop`)는 이 함수가 만든 워커 스레드 안에서 돕니다. `chat/README.md`는
이걸 "메인 스레드가 큐를 드레인한다"고 단순화해서 설명하지만, 실제 코드는 **역할이 다른 스레드 3개**로
나뉘어 있습니다.

## 3단 파이프라인 — Worker → Writer → Reader

```text
Worker 스레드(모델 개수만큼, run_llm_loop 실행)
  → merged_queue (모든 워커가 공유)
    → Writer 스레드 (_drain_to_completion, 단 1개, 항상 끝까지 돈다)
      → tee 큐
        → Reader 제너레이터 (_read_stream, HTTP 응답 쪽에서 소비, 언제든 끊겨도 됨)
```

굳이 Writer와 Reader를 분리한 이유가 주석에 명시돼 있습니다.

> "Writer-thread → reader-generator hand-off. The writer is the run's lifeline and always runs
> to completion; the reader can die freely."

**클라이언트가 브라우저를 닫아 Reader가 사라져도, Writer는 계속 돌면서 DB 저장(`_persist_model_outcome`)
까지 끝까지 해냅니다.** Reader만 있었다면 클라이언트 연결이 끊기는 순간 저장 로직까지 같이 죽어버릴
수 있었을 구조를, 큐 하나를 더 끼워 넣어 분리한 것입니다. `reader_gone` 이벤트는 Reader가 이미
사라졌을 때 `tee` 큐가 무한히 쌓이지 않도록 막는 역할만 합니다.

## Emitter — 패킷에 `model_index`만 붙여 큐에 넣는다

> 파일: `chat/emitter.py`

```python
class Emitter:
    def emit(self, packet: Packet) -> None:
        if self._drain_done is not None and self._drain_done.is_set():
            return                                          # 조기 종료 시 no-op
        tagged = Packet(placement=..., obj=packet.obj)       # model_index 태깅
        self._merged_queue.put((self._model_idx, tagged))
```

[[agent-loop]], [[llm-step]], [[tool-execution]] 어디서든 `emitter.emit(...)`만 호출하면 되고,
그 패킷이 어느 모델 것인지, 어떻게 스트리밍될지는 전혀 몰라도 됩니다 — README가 강조하는 "Emitter는
로직 없이 패킷만 취급한다"는 규칙 그대로입니다. Search API나 MCP 서버처럼 채팅 스트리밍 컨텍스트
밖에서 도구를 돌릴 때는 `NullEmitter`(그냥 버림)를 씁니다.

## Writer 스레드 — 50ms마다 취소 신호를 폴링

> 파일: `chat/process_message.py:1137-1138`, `:1520-1544`

```python
_CANCEL_POLL_INTERVAL_S: Final[float] = 0.05

model_idx, item = merged_queue.get(timeout=_CANCEL_POLL_INTERVAL_S)
# timeout에 아무것도 안 왔으면 →
if not setup.check_is_connected():        # 사용자가 정지 버튼을 눌렀는지 확인
    ... 모든 모델 부분 상태 저장 ...
    _publish(OverallStop(stop_reason="user_cancelled"))
    drain_done.set()
    return
```

새 패킷이 없는 50ms 동안마다 정지 신호(Redis)를 확인합니다. [[agent-loop]] 자체는 정지 신호를
전혀 모르는 채로 계속 돌고 있으므로, **취소는 오직 이 Writer 레벨에서만 처리**됩니다.

## `drain_done` — 워커를 강제 종료하는 게 아니라, 워커의 출력을 버리는 스위치

워커 스레드는 LLM 스트림 호출 도중이라 강제 종료가 불가능합니다(파이썬 스레드는 인터럽트 불가).
그래서 `drain_done.set()`은 스레드를 죽이는 게 아니라, **그 이후 워커가 `emitter.emit()`을 호출해도
아무 일도 안 일어나게** 만듭니다. 워커는 백그라운드에서 계속 돌다가 자기 페이스대로 끝나지만, 그
출력은 더 이상 누구에게도 전달되지 않고 조용히 버려집니다.

## 한 모델이 죽어도 다른 모델은 계속 — 완료 신호는 오직 `_MODEL_DONE`

```python
elif isinstance(item, Exception):
    _publish(StreamingError(...))   # 이 모델만 에러로 표시
    # models_remaining을 줄이지 않음! _MODEL_DONE만이 유일한 완료 신호
elif item is _MODEL_DONE:
    models_remaining -= 1
```

`_run_model`의 `finally` 블록이 성공/실패 여부와 무관하게 항상 `_MODEL_DONE`을 큐에 넣습니다
(`chat/process_message.py:1439-1441`). 그래서 예외가 나도 Writer 루프는 그 모델을 "아직 안
끝남"으로 오해하지 않고, `_MODEL_DONE`이 실제로 도착할 때만 카운트를 줄입니다 — **멀티모델
비교(N>1) 중 한 모델만 API 에러가 나도 나머지 모델의 스트림은 정상적으로 끝까지 진행**됩니다.

## 정확히 한 번만 저장 — `persist_lock` + `persisted[]`

`_persist_model_outcome(model_idx, ...)`은 세 군데에서 호출될 수 있습니다: 워커의 `finally`,
정지 버튼 처리 경로, 정상 종료 후 `_run_post_steps`. 어느 쪽이 먼저 오든 `persisted[model_idx]`
플래그를 락으로 보호해 **딱 한 번만** 실제 DB 저장이 일어나게 만듭니다 — 경쟁 상태로 같은 메시지가
두 번 저장되는 사고를 막습니다.

## `ChatStateContainer` — 모델별로 하나, 락으로 보호되는 누적 버킷

> 파일: `chat/chat_state.py`

```python
class ChatStateContainer:
    def __init__(self):
        self._lock = threading.Lock()
        self.tool_calls: list[ToolCallInfo] = []
        self.answer_tokens: str | None = None
        ...
    def set_answer_tokens(self, answer): 
        with self._lock: self.answer_tokens = answer
```

`n_models`개가 각각 별도로 생성되고, [[agent-loop]]/[[llm-step]]/[[tool-execution]]가 스트리밍
도중 계속 여기에 답변 토큰·도구 호출·인용 정보를 채워 넣습니다. 정지 버튼이 눌려도 이 컨테이너에
쌓인 **부분 상태 그대로** DB에 저장할 수 있어서, "중간에 끊긴 답변"도 있는 만큼은 보존됩니다.

## 예시 — 모델 2개 비교 중 1개가 API 에러로 죽는 경우 (N=2)

**`merged_queue` 도착 순서** (모델0=정상, 모델1=레이트리밋 에러)
```text
(0, AgentResponseStart)
(1, AgentResponseStart)
(0, AgentResponseDelta("답변을 "))
(1, RateLimitError)                ← 예외 객체 그대로 큐에 들어감
(1, _MODEL_DONE)                   ← finally에서 항상 발생
(0, AgentResponseDelta("시작합니다."))
(0, OverallStop)
(0, _MODEL_DONE)
```

**Writer의 처리**
```text
models_remaining = 2
(0,Start) → tee로 전달
(1,Start) → tee로 전달
(0,Delta) → tee로 전달
(1,Exception) → StreamingError(model_idx=1)로 tee에 전달, models_remaining 그대로(2)
(1,_MODEL_DONE) → models_remaining = 1
(0,Delta) → tee로 전달
(0,OverallStop) → tee로 전달
(0,_MODEL_DONE) → models_remaining = 0 → 루프 종료
→ 이후 두 모델 각각 _persist_model_outcome 호출 (모델0=정상 저장, 모델1=_save_errored_message)
```

**Reader(HTTP 클라이언트)가 실제로 받는 스트림** — `tee`에 들어간 순서 그대로, 모델1의 에러 패킷도
모델0의 정상 스트림과 나란히 섞여서 전달됩니다. 클라이언트 UI는 `model_index`로 어느 응답이 어느
모델 것인지 구분해서 나란히 렌더링합니다.

## 관련 노트

- [[agent-loop]] — 워커 스레드 안에서 실제로 도는 코드, `Emitter`를 인자로 받아 씀
- [[llm-step]], [[tool-execution]] — 같은 `Emitter`/`ChatStateContainer`를 계속 타고 내려가며 씀

## 정리

```text
구조:        Worker(N개, run_llm_loop) → merged_queue → Writer(1개, 항상 완주) → tee → Reader(끊겨도 무관)
분리 이유:    클라이언트 연결이 끊겨도 DB 저장(Writer)은 반드시 끝까지 실행되어야 하므로
취소 처리:    Writer가 50ms마다 폴링, agent-loop 자신은 취소를 모름
drain_done:  워커를 죽이는 게 아니라 워커의 emit()을 이후로 전부 no-op으로 만드는 스위치
에러 격리:    예외가 나도 models_remaining은 그대로, _MODEL_DONE만이 유일한 완료 신호
저장 보장:    persist_lock + persisted[]로 3개 호출 경로 중 정확히 1번만 DB 저장
상태 컨테이너: 모델별 1개, 락으로 보호되는 누적 버킷 — 정지돼도 부분 상태 그대로 저장 가능
```
