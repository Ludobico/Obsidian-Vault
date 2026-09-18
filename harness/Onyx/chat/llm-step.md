#llm-step

# LLM 추론 1회를 감싸는 곳 — `run_llm_step()`

> 파일: `chat/llm_step.py:1575` (wrapper), 실제 로직은 `:1074` `run_llm_step_pkt_generator`

[[agent-loop]]의 각 Cycle이 호출하는 함수입니다. **LLM 스트림을 실시간으로 읽으면서, 토큰이
reasoning인지 answer인지 tool_call인지 그때그때 분류해 패킷으로 내보내고**, 스트림이 끝나면
`LlmStepResult`(reasoning, answer, tool_calls, raw_answer, finish_reason)로 묶어 리턴합니다.

함수 docstring 바로 위에 이런 경고가 붙어 있습니다.

> "NOTE: DO NOT TOUCH THIS FUNCTION BEFORE ASKING YUHONG, this is very finicky and delicate
> logic that is core to the app's main functionality."

스트리밍 상태머신이라 부분 수정이 다른 케이스를 깨뜨리기 쉽다는 뜻으로 보입니다.

## 구조 — wrapper는 얇고, 진짜 로직은 제너레이터

```python
def run_llm_step(emitter, ...) -> tuple[LlmStepResult, bool]:
    step_generator = run_llm_step_pkt_generator(...)   # 진짜 로직 (제너레이터)
    while True:
        try:
            packet = next(step_generator)
            emitter.emit(packet)                        # 나오는 패킷을 그대로 스트리밍
        except StopIteration as e:
            llm_step_result, has_reasoned = e.value      # return 값은 StopIteration.value로 받음
            return llm_step_result, has_reasoned
```

`run_llm_step_pkt_generator`가 `yield Packet(...)`으로 패킷을 흘리고 마지막에
`return (LlmStepResult, has_reasoned)`을 하는 제너레이터라서, wrapper는 그걸 다 소비하며
emitter에 넘기기만 합니다. **패킷을 실시간으로 내보내면서 동시에 최종 결과값도 받아야 해서** 이런
제너레이터+`StopIteration.value` 패턴을 씁니다.

## 델타 3종류 분류 — reasoning / content / tool_calls

> 파일: `chat/llm_step.py:1307-1408`

```python
for packet in llm.stream(prompt=llm_msg_history, tools=tool_definitions, tool_choice=tool_choice, ...):
    delta = packet.choice.delta
    if delta.reasoning_content:
        # ReasoningStart(최초 1회) → ReasoningDelta 반복
    if delta.content:
        # XML 필터 통과 후 → AgentResponseStart(최초 1회) → AgentResponseDelta 반복
    if delta.tool_calls:
        # 진행 중이던 reasoning을 ReasoningDone으로 닫고, 델타를 tool_call 맵에 누적
```

세 채널이 **한 스트림 안에서 순서 없이 섞여 들어올 수 있어서**, `reasoning_start`/`answer_start`
플래그로 "이미 Start 패킷을 보냈는지"를 추적합니다. 예를 들어 reasoning 도중 tool_calls 델타가
오면 `_close_reasoning_if_active()`가 즉시 `ReasoningDone`을 내보내고 `turn_index`를 1 증가시킵니다
— 이게 [[agent-loop]] 노트에서 언급한 "reasoning + tool_call이 백엔드로는 한 Cycle이지만 프론트엔드엔
2개의 turn_index로 렌더링되는" 이유입니다.

## 답변 텍스트에 섞여나온 XML 도구호출을 실시간으로 걸러낸다

> 파일: `chat/llm_step.py:89-148` (`_XmlToolCallContentFilter`)

일부 모델은 tool calling API 대신 `<function_calls><invoke name="...">...</invoke></function_calls>`
같은 XML을 answer 텍스트 안에 그대로 뱉습니다. 이게 사용자에게 그대로 스트리밍되면 안 되므로,
`_XmlToolCallContentFilter`가 **청크 경계에 걸쳐 잘린 `<function_calls` 마커까지 버퍼링**하면서
블록 전체를 제거한 뒤에만 나머지를 `_emit_content_chunk`로 흘려보냅니다. (진짜 도구 호출로의 변환은
여기서 안 하고, [[agent-loop|_try_fallback_tool_extraction]]가 스트림 종료 후 별도로 처리합니다.)

## 도구 호출 조립 — 델타 조각을 모아 완성된 `ToolCallKickoff`로

> 파일: `chat/llm_step.py:355-422`

OpenAI 계열 스트리밍은 tool_call을 한 번에 안 주고 `index`별로 `id`/`name`/`arguments`를 조각조각
보냅니다. `_update_tool_call_with_delta`가 `id_to_tool_call_map[index]`에 계속 append하고,
스트림이 끝난 뒤 `_extract_tool_call_kickoffs`가 `id`와 `name`이 둘 다 채워진 것만 골라
`ToolCallKickoff`로 변환합니다.

## 스트림이 끝났는데 답이 비어있으면 — raw 답변으로 복구

> 파일: `chat/llm_step.py:1453-1498`

```python
if (
    tool_choice != ToolChoiceOptions.REQUIRED
    and not tool_calls
    and not accumulated_answer.strip()
    and accumulated_raw_answer.strip()
    and not _looks_like_xml_tool_call_payload(accumulated_raw_answer)
):
    accumulated_answer = accumulated_raw_answer   # 가공된 답이 비었으면 원본으로 대체
```

인용 프로세서가 `"[123456789012345]"` 같은 매핑 안 되는 숫자 브래킷을 인용으로 오인해 통째로
삭제해버리는 경우처럼, **가공 과정에서 답이 실수로 다 지워질 수 있어서** 원본(raw)을 보존해뒀다가
복구합니다. 단, XML 도구호출처럼 보이면 복구하지 않고 [[agent-loop]]의 폴백 추출로 넘깁니다 —
그렇지 않으면 사용자에게 raw XML 마크업이 그대로 노출되기 때문입니다.

## 예시 입력/출력

**입력**: `history=[S, U1]`, `tool_calls`가 없는 평범한 모델의 `llm.stream()`이 아래 순서로 델타를 냄

```text
delta 1: content="검색해서 확인해볼게요."
delta 2: tool_calls=[{index:0, id:"call_1", function:{name:"web_search"}}]
delta 3: tool_calls=[{index:0, function:{arguments:'{"query":'}}]
delta 4: tool_calls=[{index:0, function:{arguments:'"bitcoin price"}'}}]
delta 5: finish_reason="tool_calls"
```

**`run_llm_step` 도중 emitter로 나가는 패킷 (순서대로)**

```text
AgentResponseStart(...)
AgentResponseDelta(content="검색해서 확인해볼게요.")
ToolCallKickoff 관련 argument-delta 패킷들 (maybe_emit_argument_delta, id=call_1)
```

**스트림 종료 후 함수의 리턴값**

```python
(
    LlmStepResult(
        reasoning=None,
        answer="검색해서 확인해볼게요.",
        tool_calls=[ ToolCallKickoff(tool_call_id="call_1", tool_name="web_search",
                                      tool_args={"query": "bitcoin price"}) ],
        raw_answer="검색해서 확인해볼게요.",
        finish_reason="tool_calls",
    ),
    has_reasoned=False,
)
```

이 리턴값이 그대로 [[agent-loop]]의 `run_llm_step()` 호출부로 돌아가서, `tool_calls`가 있으니
`run_tool_calls()`로 넘어가는 다음 단계가 이어집니다.

**reasoning 모델이었다면** delta 1 대신 `delta.reasoning_content` 조각들이 먼저 오고
(`ReasoningStart` → `ReasoningDelta`×N), tool_calls 델타가 도착하는 순간 `_close_reasoning_if_active()`가
`ReasoningDone`을 내보내며 `has_reasoned=True`로 리턴됩니다 — [[agent-loop]]에서 이 값이
`reasoning_cycles`를 1 증가시키는 데 쓰입니다.

## 정리

```text
역할:        llm.stream()의 원시 델타를 reasoning/answer/tool_call 3채널로 분류해 패킷 스트리밍
구조:        wrapper(run_llm_step) — 얇음 / 제너레이터(run_llm_step_pkt_generator) — 진짜 로직
채널 경계:    reasoning_start/answer_start 플래그로 Start/Done 패킷 중복 방지
XML 오염 방지: _XmlToolCallContentFilter가 <function_calls> 블록을 답변 텍스트에서 실시간 제거
도구 조립:    스트리밍 델타를 index별로 누적 → 스트림 종료 후 완성된 ToolCallKickoff로 변환
빈 답 복구:   가공 후 답이 비면 raw_answer로 대체 (단 XML tool-call처럼 보이면 상위 폴백에 위임)
리턴:         (LlmStepResult, has_reasoned) — has_reasoned는 상위 루프의 turn_index 증가에 사용
```
