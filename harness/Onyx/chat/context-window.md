#context-window

# 히스토리를 토큰 예산 안으로 자르는 곳 — `construct_message_history()`

> 파일: `chat/llm_loop.py:408-629` (`construct_message_history`), `chat/token_budget.py` (출력 토큰 예산)

[[agent-loop]]가 매 사이클 호출하는 함수입니다. `simple_chat_history`(영구 누적 리스트)를 그대로
LLM에 보내는 게 아니라, **이번 사이클의 토큰 예산 안에 들어오도록 매번 새로 잘라서** 리턴합니다.

## 예산 계산 — 시스템/커스텀/프로젝트/리마인더부터 빼고 남는 게 히스토리 몫

```python
history_token_budget = available_tokens
history_token_budget -= system_prompt.token_count if system_prompt else 0
history_token_budget -= custom_agent_prompt.token_count if custom_agent_prompt else 0
history_token_budget -= project_messages_tokens
history_token_budget -= reminder_message.token_count if reminder_message else 0
```

`available_tokens`는 [[agent-loop]]에서 `max(0, available_tokens - tool_token_budget)`로 이미
도구 정의 토큰까지 뺀 값이 들어옵니다. 즉 **"도구 목록 → 시스템 프롬프트 → 커스텀 에이전트 →
프로젝트 파일 → 리마인더"를 다 뺀 나머지가 대화 히스토리에 쓸 수 있는 양**이라는 우선순위가
코드 순서 그대로 반영돼 있습니다.

## 히스토리를 3부분으로 쪼갠다 — "마지막 사용자 메시지"가 기준점

```text
history_before_last_user   ← 자를 수 있는 부분 (오래된 것)
last_user_message          ← 반드시 포함
messages_after_last_user   ← 반드시 포함 (이번 Turn에서 이미 실행된 TC/TR들)
```

`last_user_message`와 그 뒤(`messages_after_last_user`, 즉 이번 Turn에서 나온 TC/TR)는 **절대
자르지 않습니다.** LLM API가 선형 히스토리를 요구하는 이상, 지금 진행 중인 Turn의 도구 호출/응답
체인을 중간에 끊을 수는 없기 때문입니다. 이 둘의 토큰 합이 예산을 넘으면 아예 `ValueError`를 던지고
실패시킵니다 — 자를 방법이 없는 상황이라 조용히 실패하지 않습니다.

## 오래된 히스토리부터, 최신순으로 채우다가 예산 넘으면 즉시 중단

```python
for msg in reversed(history_before_last_user):     # 최신 → 과거 순회
    if current_token_count + msg_tokens <= remaining_budget:
        truncated_history_before.insert(0, msg)      # 채택, 앞쪽에 다시 끼워넣음
        current_token_count += msg_tokens
    else:
        break                                         # 여기서 멈춤 — 더 과거 것도 시도 안 함
```

**"이 메시지가 안 맞으면 그보다 더 오래된 메시지도 안 맞다고 가정하고 멈춥니다"** (continue가 아니라
break). 오래된 메시지 중 유독 짧은 게 있어도 건너뛰고 챙기지 않습니다 — 최근 대화 흐름을 시간순으로
온전히 보존하는 것이 조각난 과거 조각들을 더 넣는 것보다 낫다는 설계입니다.

## 잘린 히스토리에 파일이 있었다면 — "잊혀진 파일" 메타데이터로 대체

잘려나간 메시지 중 `file_id`가 있던 것들은 그냥 사라지지 않고, `FileReaderTool`이 붙어있는 경우
`forgotten_files_message`(파일명 + `file_id`만 있는 가벼운 안내 메시지)로 대체됩니다. 이 메타데이터
메시지도 토큰을 차지하므로, 넣고 나서 예산을 넘으면 **이미 채택했던 히스토리를 추가로 더 쫓아냅니다**
(`while truncated_history_before and current_token_count > remaining_budget: evicted = ...pop(0)`).
잘린 파일 자체를 위한 공간을 확보하려고 재차 과거 메시지를 희생시키는 2단계 truncation인 셈입니다.

## 최종 조립 순서

```text
[system] + [잘린 과거 히스토리] + [custom_agent] + [project_files] + [forgotten_files]
         + [last_user_message] + [messages_after_last_user] + [reminder]
```

`chat/README.md`에 문서화된 순서 그대로입니다 — 커스텀 에이전트/프로젝트 파일이 "마지막 사용자
메시지 바로 앞"에 항상 오도록 고정된 것도 이 조립 순서 때문입니다.

## 고아 도구응답 제거 — `_drop_orphaned_tool_call_responses`

> 파일: `chat/llm_loop.py:632-663`

방금 본 newest-first truncation에는 흥미로운 부작용이 있습니다. `TR`(도구 응답)은 자신의 `TC`
(도구 호출)보다 **항상 더 최신**이므로, 순회 순서상 `TR`이 먼저 채택되고 그 다음 `TC` 차례에서
예산이 부족해 `break`가 걸릴 수 있습니다 — **TC 없이 TR만 살아남는 상황**이 생깁니다. Ollama 같은
일부 프로바이더는 이런 히스토리를 "unexpected tool call id" 에러로 거부하므로, 이 함수가 최종
결과에서 `tool_call_id`가 앞선 ASSISTANT 메시지에 없는 `TOOL_CALL_RESPONSE`를 걸러냅니다.

## 입력 토큰과 출력 토큰은 별도 예산 — `ChatTokenBudget`

> 파일: `chat/token_budget.py`

```python
@dataclass(frozen=True)
class ChatTokenBudget:
    input_tokens: int          # 안전마진 뺀 입력 예산 (위 available_tokens로 쓰임)
    max_output_tokens: int | None
    context_tokens: int | None
    safety_tokens: int

    def output_allowance(self, estimated_input_tokens: int) -> int | None:
        available_output_tokens = self.context_tokens - self.safety_tokens - estimated_input_tokens
        if available_output_tokens < min(self.max_output_tokens, max(1, GEN_AI_NUM_RESERVED_OUTPUT_TOKENS)):
            return None
        return min(self.max_output_tokens, available_output_tokens)
```

`resolve_chat_token_budget(llm)`이 모델의 `max_input_tokens`에 `GEN_AI_INPUT_TOKEN_SAFETY_MARGIN`
만큼 안전마진을 뗀 값을 `input_tokens`(=`available_tokens`)로 씁니다. **입력 예산은 Turn 시작 시
한 번 정해지지만, 출력 허용량(`output_allowance`)은 사이클마다 실제로 조립된 히스토리 토큰 수를
다시 넣어 재계산**됩니다 — 컨텍스트 윈도우 전체(`context_tokens`)에서 안전마진과 이번 요청의 실제
입력 토큰을 뺀 나머지가 이번 사이클에 모델이 답할 수 있는 최대 길이입니다.

## 예시 입력/출력 — 히스토리 truncation

`history_token_budget = 1000`(설명을 위해 작게 잡음), `history_before_last_user`(오래된 순)와
`last_user_message`가 아래와 같다고 하겠습니다.

```text
history_before_last_user = [ U0(300 tok, file_id 없음), A0(600 tok) ]
last_user_message        = U1(150 tok)
messages_after_last_user = []  (이번 Turn 첫 사이클이라 아직 TC/TR 없음)
```

**1단계 — 필수 구간 먼저 확보**
```text
required_tokens = 150(U1) + 0(after) = 150   → 1000 이내이므로 통과
remaining_budget = 1000 - 150 = 850
```

**2단계 — `history_before_last_user`를 최신순(A0 → U0)으로 채택**
```text
A0(600): 0 + 600 = 600 ≤ 850           → 채택, current_token_count = 600
U0(300): 600 + 300 = 900 > 850         → 탈락, 여기서 break (U0보다 오래된 것도 더 없음)
```

**결과 — 최종 조립**
```text
[S] + [A0] + [U1] + [] + [R]
```
가장 오래된 사용자 메시지(U0)는 잘려나가고 그 답변(A0)만 히스토리에 남습니다. `U0`에 `file_id`가
있었다면 `dropped_file_ids`에 잡혀 `forgotten_files_message`로 대체됐을 자리입니다.

## 관련 노트

- [[agent-loop]] — 이 함수를 매 사이클 호출하는 상위 루프, `available_tokens` 계산 지점
- [[prompt-assembly]] — 여기서 만들어진 `system_prompt`가 이 함수의 첫 번째 인자로 들어감

## 정리

```text
자르는 기준:  마지막 사용자 메시지 이전 = 자를 수 있음 / 그 이후(현재 Turn) = 무조건 보존
순회 방향:    최신 → 과거로 채우다가 처음 안 맞는 순간 break (건너뛰고 더 뒤짐 없음)
파일 손실 대응: 잘린 file_id는 forgotten_files_message로 대체 (2차 truncation 발생 가능)
부작용:       TR이 TC보다 먼저 채택되어 TC 없는 TR만 남을 수 있음 → _drop_orphaned_tool_call_responses가 정리
예산 2종:     입력(input_tokens, Turn 시작 시 고정) / 출력(output_allowance, 사이클마다 재계산)
```
