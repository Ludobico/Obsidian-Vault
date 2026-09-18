#tool-execution

# 도구가 실제로 실행되는 곳 — `run_tool_calls()`

> 파일: `tools/tool_runner.py:232-453`

[[agent-loop]]가 `llm_step_result.tool_calls`를 받은 다음 넘기는 곳입니다. 도구를 그냥 하나씩
순서대로 실행하는 게 아니라 **병합 → 필터링 → 인용번호 사전 할당 → 병렬 실행 → 실패 격리**까지
5단계를 거칩니다.

## 1단계 — 같은 검색류 도구를 여러 번 부르면 하나로 합친다

> 파일: `tools/tool_runner.py:56-115` (`_merge_tool_calls`)

```python
MERGEABLE_TOOL_FIELDS: dict[str, str] = {
    SearchTool.NAME: "queries",
    WebSearchTool.NAME: "queries",
    OpenURLTool.NAME: "urls",
}
```

LLM이 한 사이클에 `internal_search`를 3번 따로 호출해도(예: 서로 다른 쿼리 3개), 실제로는
**쿼리 리스트를 합쳐 단 한 번만 실행**합니다. 검색 백엔드에 3번 왕복하는 대신 1번의 배치 호출로
줄이려는 최적화입니다. `open_url`도 URL 리스트를 같은 방식으로 병합합니다. 그 외 도구는 병합 없이
각각 그대로 실행됩니다.

## 2단계 — 알 수 없는 도구 / 과다 호출 컷

```python
if tool_call.tool_name not in tools_by_name:
    logger.warning(...)   # 조용히 버림, 에러로 죽지 않음
if max_concurrent_tools is not None:
    filtered_tool_calls = filtered_tool_calls[:max_concurrent_tools]
```

바인딩되지 않은 도구를 LLM이 환각으로 호출해도 시스템이 죽지 않고 그냥 무시합니다. `max_concurrent_tools`가
있으면 그 이후 호출은 **큐에 쌓지도 않고 드롭**합니다.

## 3단계 — 인용 번호를 도구별로 100개씩 미리 구간 예약

```python
starting_citation_num = next_citation_num
for tool_call in filtered_tool_calls:
    ...
    override_kwargs = SearchToolOverrideKwargs(starting_citation_num=starting_citation_num, ...)
    starting_citation_num += 100
```

도구들이 **병렬로 실행**되기 때문에, 실행이 끝난 순서대로 인용 번호를 매기면 두 도구가 동시에
"다음 번호는 5번"이라고 계산해서 충돌할 수 있습니다. 그래서 실행 전에 미리 도구별로 100개씩
떼어 할당해버립니다(한 도구 호출이 100개 넘는 문서를 인용할 일은 없다고 가정한 여유값).

## 4단계 — 도구별 override_kwargs 분기

같은 `run()` 시그니처를 쓰지만, 도구마다 필요한 부가 정보가 달라서 `isinstance` 분기로 채워줍니다.

```text
SearchTool      → message_history, user_memory_context, skip_query_expansion 등
WebSearchTool   → starting_citation_num만
OpenURLTool     → citation_mapping(URL→기존 인용번호 역매핑), url_snippet_map
PythonTool      → chat_files (코드 인터프리터에 업로드할 파일들)
MemoryTool      → user_name/email/role + existing_memories (메모리 문구 일관성 위해, [[memory]] 참고)
CodingAgentTool → 빈 kwargs (자체 서브 루프가 모든 상태를 내부에서 관리)
```

## 5단계 — 병렬 실행, 실패는 격리

> 파일: `tools/tool_runner.py:118-229` (`_safe_run_single_tool`)

```python
tool_run_results = run_functions_tuples_in_parallel(
    functions_with_args, allow_failures=True, max_workers=max_concurrent_tools,
    timeout=TOOL_EXECUTION_TIMEOUT_SECONDS,   # 10분
)
```

`_safe_run_single_tool`이 도구 하나당 예외를 3단으로 구분해서 잡습니다.

```text
ToolCallException     예상된 실패(잘못된 입력, API 실패 등) → e.llm_facing_message 그대로 LLM에 전달
ToolExecutionException 예상 못한 실행 중 에러 → 제네릭 에러 메시지 + emit_error_packet이면 패킷도 emit
Exception (그 외)      완전 예상 못한 에러 → 제네릭 에러 메시지, 프로세스는 안 죽음
```

**한 도구가 죽어도 나머지 도구 결과와 이번 사이클 전체가 죽지 않습니다.** `allow_failures=True` +
이 3단 try/except 덕분에, 실패한 도구는 그냥 "Tool failed with error: ..."라는 텍스트를 담은
정상적인 `ToolResponse`가 되어 [[agent-loop]]로 돌아갑니다 — [[agent-loop]] 노트의 "도구 실행이
통째로 실패하면" 케이스는 **모든** 도구가 실패해서 `tool_responses`가 아예 빈 경우만 해당하고,
일부만 실패하는 경우는 이 레이어에서 이미 흡수됩니다.

## 예시 입력/출력

**입력** — 한 사이클에서 LLM이 검색을 2번, 웹서치를 1번 호출:

```python
tool_calls = [
    ToolCallKickoff(tool_name="internal_search", tool_args={"queries": ["Q3 로드맵"]}),
    ToolCallKickoff(tool_name="internal_search", tool_args={"queries": ["Q3 예산"]}),
    ToolCallKickoff(tool_name="web_search",      tool_args={"queries": ["온보드 최신 뉴스"]}),
]
next_citation_num = 1
```

**병합 후 (`_merge_tool_calls`)**
```text
internal_search: queries=["Q3 로드맵", "Q3 예산"]   ← 2개 호출이 1개로 합쳐짐
web_search:      queries=["온보드 최신 뉴스"]        ← 그대로
```

**인용 번호 할당 + 병렬 실행**
```text
internal_search → starting_citation_num=1   (실행 후 next는 101)
web_search       → starting_citation_num=101 (실행 후 next는 201)
```
두 도구는 스레드풀에서 동시에 실행되고, `internal_search`가 문서 5개를 찾았다면 그 문서들은
1~5번, `web_search`가 3개를 찾았다면 101~103번을 받습니다 — 완료 순서와 무관하게 번호가 안 겹칩니다.

**만약 web_search가 타임아웃으로 실패했다면**
```python
ToolResponse(
    rich_response=None,
    llm_facing_response="Tool failed with error: <원인 메시지>",
    tool_call=<원래 web_search ToolCallKickoff>,
)
```
이 응답도 그대로 `tool_responses` 리스트에 포함되어 반환되고, `internal_search` 결과는 정상적으로
같이 돌아갑니다.

**최종 리턴**
```python
ParallelToolCallResponse(
    tool_responses=[<internal_search 결과>, <web_search 실패 응답>],
    updated_citation_mapping={1: doc_a, 2: doc_b, ..., 101: ...},  # 실제로 찾은 문서만 채워짐
)
```

## 관련 노트

- [[agent-loop]] — 이 함수를 호출하는 상위 루프, 전체 실패(모든 tool_responses가 빔) 시의 재시도 처리
- [[memory]] — `MemoryTool`에 전달되는 override_kwargs가 왜 사용자 이름/이메일까지 필요한지

## 정리

```text
병합:       SearchTool/WebSearchTool(queries), OpenURLTool(urls) — 같은 도구 중복 호출을 1회로
필터링:     바인딩 안 된 도구 이름은 조용히 무시, max_concurrent_tools 넘으면 드롭(큐잉 없음)
인용번호:   병렬 실행 전에 도구별로 100개씩 미리 구간 예약 → 완료 순서와 무관하게 충돌 없음
실행:       스레드풀 병렬, 10분 타임아웃, allow_failures=True
실패 처리:  ToolCallException/ToolExecutionException/그외 3단 분리, 어느 것도 전체 배치를 죽이지 않음
```
