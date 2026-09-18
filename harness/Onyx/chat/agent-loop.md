#agent-loop

# 에이전트 루프가 도는 곳 — `run_llm_loop()`

> 파일: `chat/llm_loop.py:813` (함수 시작), 사이클 본문은 `:949-1469`

Onyx의 "하네스"는 별도 프레임워크가 아니라 **하나의 함수 안 while문**입니다. `chat/README.md`가
개념을 이렇게 정의합니다.

```text
Turn  — 사용자가 메시지 1개를 보내고 에이전트가 최종 답변을 낼 때까지의 전체 처리
Cycle — 그 안에서 LLM 추론 1회 (도구 호출 여부와 무관)
```

`run_llm_loop()` 하나가 Turn 전체를 담당하고, 내부의 `for llm_cycle_count in range(MAX_LLM_CYCLES)`
가 Cycle 하나하나에 대응합니다.

## 사이클 하나의 흐름

매 반복마다 이 순서로 돕니다 (`llm_loop.py:949-1454`).

```text
1. tool_choice 결정        AUTO / REQUIRED / NONE 중 하나 (아래 참고)
2. 시스템 프롬프트 조립     build_system_prompt() → 사이클마다 새로 만듦 (캐시는 프롬프트 캐싱으로 처리)
3. 리마인더 선택           select_reminder_text() — 인용/파일/이미지 상황에 따라 다른 문구
4. 히스토리 자르기         construct_message_history() — 토큰 예산 안으로 truncate
5. run_llm_step() 호출     실제 LLM 추론 1회, reasoning/answer/tool_calls를 받음
6. 도구 실행               run_tool_calls() — tool_calls를 병렬 실행
7. 결과를 히스토리에 반영   ASSISTANT(tool_calls) + TOOL_CALL_RESPONSE 메시지로 append
8. 종료조건 체크           tool_calls가 없으면 break, 있으면 다음 사이클로
```

**한 사이클 = 프롬프트를 처음부터 다시 조립 + 히스토리를 다시 자르는 것**이라는 점이 특이합니다.
이전 사이클의 프롬프트를 재사용하지 않고, `tool_choice`/`reminder`/`should_cite_documents` 같은
그 사이클의 상태를 반영해 매번 새로 만듭니다.

## tool_choice가 사이클마다 바뀐다 — AUTO / REQUIRED / NONE

```python
if forced_tool_id:
    final_tools = [tool for tool in tools if tool.id == forced_tool_id]
    tool_choice = ToolChoiceOptions.REQUIRED
    forced_tool_id = None                      # 딱 1회만 강제
elif out_of_cycles or ran_image_gen:
    tool_choice = ToolChoiceOptions.NONE        # 도구 자체를 안 줌 → 답변 강제
    final_tools = []
else:
    tool_choice = ToolChoiceOptions.AUTO
    final_tools = tools
```

- `forced_tool_id`는 최초 1회 특정 도구를 강제로 부르게 하는 경우(예: 사용자가 특정 액션을 명시적으로
  선택)에 쓰이고, 쓰이자마자 `None`으로 리셋됩니다 — **강제는 정확히 한 사이클만** 유효합니다.
- `out_of_cycles`(마지막 사이클)에는 도구 목록 자체를 빈 배열로 넘겨서 **모델이 물리적으로 도구를
  호출할 수 없게** 만듭니다. "이제 그만 답하라"는 프롬프트 지시가 아니라 API 레벨의 강제입니다.

## MAX_LLM_CYCLES 기본값 6 — 왜 6인가

> 파일: `chat/llm_loop.py:306-313`, `configs/chat_configs.py:9-12`

```python
# Default 6 covers the common search → open_url pattern:
# Cycle 1: Calls web_search for something
# Cycle 2: Calls open_url for some results
# Cycle 3: Calls web_search for some other aspect of the question
# Cycle 4: Calls open_url for some results
# Cycle 5: Maybe call open_url for some additional results or because last set failed
# Cycle 6: No more tools available, forced to answer
MAX_LLM_CYCLES: int = int(os.environ.get("MAX_LLM_CYCLES") or 6)
```

임의의 숫자가 아니라 **"검색 → 링크 열람"을 2번 왕복하는 실제 사용 패턴**을 역산해서 나온 값입니다.
도구 호출이 많이 필요한 MCP를 연결할 때는 환경변수로 늘리라고 주석에 명시돼 있습니다.

## 종료 조건 — 도구 호출이 없으면 끝

```python
if not llm_step_result.tool_calls or len(llm_step_result.tool_calls) == 0:
    break
```

루프의 유일한 정상 종료 조건입니다. `MAX_LLM_CYCLES`에 도달하는 것은 사실 이 조건의 특수 케이스로
수렴합니다 — 마지막 사이클엔 `tool_choice=NONE`이라 애초에 tool_calls가 나올 수 없기 때문입니다.

## 약한 모델을 위한 폴백 — 텍스트에서 도구 호출 추출

> 파일: `chat/llm_loop.py:216-303` (`_try_fallback_tool_extraction`)

네이티브 tool calling을 지원하지 않거나 품질이 낮은 모델이 도구 호출을 XML/텍스트로 뱉어버리는
경우, `answer` → `raw_answer` → `reasoning` 순서로 훑어서 도구 호출을 다시 파싱합니다. 단,

```python
if fallback_extraction_attempted:
    return llm_step_result, False
```

**한 Turn당 딱 한 번만 시도**합니다. 계속 실패하는 모델을 상대로 무한히 재시도하며 루프를 도는
사고를 막기 위한 안전장치입니다.

## 도구 실행이 통째로 실패하면 — 에러를 감추지 않고 재시도 기회를 줌

```python
if tool_calls and not tool_responses:
    failure_messages = create_tool_call_failure_messages(tool_calls, token_counter)
    simple_chat_history.extend(failure_messages)
    continue
```

도구 실행이 전멸하면 그 사이클을 그냥 버리는 게 아니라, **실패 사실을 히스토리에 메시지로 남기고
다음 사이클로 넘깁니다.** LLM이 다음 추론에서 실패를 인지하고 다른 방식으로 재시도하거나 포기하고
답변하도록 유도하는 구조입니다.

## 도구 종류별로 사이클 정책이 갈린다

> 파일: `tools/built_in_tools.py:47-52`

```python
STOPPING_TOOLS_NAMES: list[str] = [ImageGenerationTool.NAME]
CITEABLE_TOOLS_NAMES: list[str] = [SearchTool.NAME, WebSearchTool.NAME, OpenURLTool.NAME]
```

- **이미지 생성 도구가 호출되면** `ran_image_gen = True` → 위 tool_choice 로직에 의해 **다음
  사이클은 강제로 `NONE`.** 이미지를 만든 다음 또 다른 도구를 부르게 두지 않고 바로 마무리 답변으로
  넘어가게 하는 정책입니다.
- **인용 가능한 도구(검색류)가 한 번이라도 호출되면** `should_cite_documents = True`가 Turn이 끝날
  때까지 유지되어, 이후 모든 사이클의 시스템 프롬프트/리마인더에 인용 지시가 계속 포함됩니다.

## 예시 Trace — "시세 알려주고 뉴스도 요약해줘"가 3사이클로 끝나는 경우

`chat/README.md`의 표기법을 그대로 씁니다 (S=시스템, U=사용자, TC=도구호출, TR=도구응답, R=리마인더, A=최종답변).

> **U1**: "오늘 비트코인 시세 알려주고, 관련 최신 뉴스 하나 요약해줘"

**Cycle 0** — `tool_choice=AUTO`
```text
LLM 입력:  [S] [U1]
LLM 출력:  tool_calls = [ web_search(query="bitcoin price today") ]
도구 실행: web_search → 검색결과 목록
플래그:    has_called_search_tool=True, just_ran_web_search=True
영구 반영: simple_chat_history += [TC1, TR1]
종료?      tool_calls가 있었으므로 계속
```

**Cycle 1** — 리마인더가 이번 사이클에만 새로 계산됨 (`just_ran_web_search=True` + `open_url` 보유 → `OPEN_URL_REMINDER`)
```text
LLM 입력:  [S] [U1] [TC1] [TR1] [R: "링크가 있으면 open_url로 열어보세요"]
LLM 출력:  tool_calls = [ open_url(url="https://news.example/btc-article") ]
도구 실행: open_url → 기사 본문
플래그:    just_ran_web_search=False로 리셋(open_url은 WebSearchTool이 아님),
           should_cite_documents=True (open_url도 CITEABLE_TOOLS_NAMES)
영구 반영: simple_chat_history += [TC2, TR2]      ※ R은 히스토리에 저장되지 않음
종료?      tool_calls가 있었으므로 계속
```

**Cycle 2** — `should_cite_documents=True`가 유지되므로 이번엔 인용 리마인더
```text
LLM 입력:  [S] [U1] [TC1] [TR1] [TC2] [TR2] [R: "답변에 인용을 포함하세요"]
LLM 출력:  answer = "현재 비트코인 시세는 약 $X입니다[[1]]. 관련 기사 요약: ...[[2]]", tool_calls = []
종료?      tool_calls가 비었으므로 break → OverallStop emit
```

```text
핵심: simple_chat_history는 TC/TR만 영구 누적, R(리마인더)은 매 사이클 휘발성으로 새로 계산됨
      종료는 MAX_LLM_CYCLES(=6) 도달이 아니라 "LLM이 도구를 안 부른 순간" — 이 예시는 3사이클 만에 자연 종료
```

## 관련 노트

- [[memory]] — `inject_memories_in_prompt` 플래그가 이 루프의 매 사이클 프롬프트 조립에 관여
- [[prompt-assembly]] — `build_system_prompt()`가 이 루프 안에서 사이클마다 호출되는 지점
- `[[context-window]]` — `construct_message_history()`의 히스토리 truncation 로직 (다음 노트에서 다룸)

## 정리

```text
루프 단위:    run_llm_loop() 1회 = Turn 1개, for문 반복 1회 = Cycle 1개
종료 조건:    tool_calls가 없으면 break (MAX_LLM_CYCLES는 이 조건을 강제로 만족시키는 안전판)
기본 한도:    6 사이클 (검색→링크열람 2왕복 기준, env로 조정 가능)
tool_choice:  AUTO(기본) / REQUIRED(1회 강제 후 리셋) / NONE(마지막 사이클, 이미지생성 직후)
폴백:         텍스트 기반 도구 호출 추출 — Turn당 1회 한정
도구 실패:    전멸 시 continue로 다음 사이클 재시도 유도 (즉시 에러 아님)
도구별 훅:    이미지생성 → 다음 사이클 NONE 강제 / 검색류 → 인용 지시 지속
```
