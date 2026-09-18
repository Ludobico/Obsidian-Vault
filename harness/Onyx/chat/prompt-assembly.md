#prompt-design

# 프롬프트가 실제로 조립되는 곳 — `build_system_prompt()`

> 파일: `chat/prompt_utils.py:250+`

이전 노트들에서 본 문자열 상수(`chat_prompts.py`, `tool_prompts.py`, `user_info.py`)들이 실제로
어떤 순서로 이어붙여지는지가 이 함수 하나에 담겨 있습니다.

```text
1. apply_prompt_placeholders()      base_system_prompt의 {{CURRENT_DATETIME}} 등 치환
2. _build_user_information_section() "# User Information" 블록 추가
3. (필요시) REQUIRE_CITATION_GUIDANCE 추가  — placeholder가 없었을 때만 폴백으로
4. 바인딩된 도구 종류에 따라 TOOL_SECTION_HEADER + 관련 안내문만 선택적으로 추가
```

## 1단계 — 플레이스홀더 치환 (`apply_prompt_placeholders`)

> 파일: `prompts/prompt_utils.py:116-152`

```python
def apply_prompt_placeholders(prompt_str, *, datetime_aware=False, append_datetime_if_aware=False,
                               should_cite_documents=False, ...) -> tuple[str, bool]:
    prompt_str = replace_reminder_tag(prompt_str)
    original_prompt = prompt_str
    prompt_str = replace_current_datetime_tag(prompt_str, full_sentence=False, include_day_of_week=True)
    if prompt_str == original_prompt and append_datetime_if_aware and datetime_aware:
        # 프롬프트에 {{CURRENT_DATETIME}} 태그가 아예 없었으면, 뒤에 날짜를 그냥 덧붙임
        prompt_str += ADDITIONAL_INFO.format(...)
    return replace_citation_guidance_tag(...)
```

"플레이스홀더가 있으면 그 자리에 치환하고, 없으면 뒤에 덧붙인다"는 **하위 호환 폴백 전략**입니다 —
관리자가 커스텀 프롬프트를 쓰면서 `{{CURRENT_DATETIME}}` 태그를 안 넣었어도 날짜 정보가 아예
빠지지는 않게 합니다.

## 2단계 — 회사 컨텍스트 (`get_company_context`)

> 파일: `prompts/prompt_utils.py:198-218`

```python
def get_company_context() -> str | None:
    workspace_settings = load_settings()
    if not workspace_settings.company_name and not workspace_settings.company_description:
        return None
    ...  # COMPANY_NAME_BLOCK / COMPANY_DESCRIPTION_BLOCK 조합
```

관리자가 워크스페이스 설정에서 회사명/회사 설명을 입력해두면 모든 프롬프트에 자동으로 섞입니다.

## 3단계 — User Information 섹션

> 파일: `chat/prompt_utils.py:181+`, docstring 그대로:

> "'# User Information' sub-sections, in order: **Basic Info → Organization Profile → Team Info →
> Language → Preferences → Memories**."

```text
## Basic Information     — 이름/이메일/역할
## Organization Profile   — IdP(회사 로그인 시스템)에서 가져온 국가/부서 등 디렉토리 정보
## Team Information       — 사용자가 속한 팀 설명
## Language               — UI 언어 → 답변 언어 힌트
## User Preferences       — 사용자 선호(문단 형태)
## User Memories          — 저장된 메모리 목록 (자세한 내용은 [[03-memory|메모리 노트]] 참고)
```

**순서가 고정돼 있고, 메모리는 항상 맨 마지막입니다.** 앞쪽(이름/역할/조직/팀/언어)은 매 요청마다
거의 안 바뀌는 값이고, 메모리만 사용자가 대화 중 계속 늘려가는 값이라는 걸 감안한 배치로 보입니다.

## 4단계 — 도구별 안내문은 "바인딩된 도구만" 골라서 추가

```python
has_web_search = any(isinstance(tool, WebSearchTool) for tool in tools)
has_internal_search = any(isinstance(tool, SearchTool) for tool in tools)
has_memory = any(isinstance(tool, MemoryTool) for tool in tools)
...
if has_memory or include_all_guidance:
    tool_guidance_sections.append(MEMORY_GUIDANCE)
```

`tool_prompts.py`의 각 안내문(`INTERNAL_SEARCH_GUIDANCE`, `MEMORY_GUIDANCE` 등)은 **그 도구가
실제로 이 에이전트에 바인딩돼 있을 때만** 프롬프트에 들어갑니다 — 안 쓰는 도구의 설명으로 프롬프트
용량을 낭비하지 않는 구조입니다.

## 정리

```text
조립 순서: base(플레이스홀더 치환) → 회사 컨텍스트 → 사용자 정보(이름~팀~언어~선호~메모리 순)
          → 인용 가이드(폴백) → 바인딩된 도구 안내문만 선택적으로
```

각 상수의 실제 원문과 한국어 번역은 [[prompt-strings]] 참고.

