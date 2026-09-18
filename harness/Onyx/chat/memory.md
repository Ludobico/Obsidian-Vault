#memory

# 메모리 시스템

## 저장 구조 — `UserMemoryContext`

> 파일: `db/memory.py`

```python
MAX_MEMORIES_PER_USER = 10   # 사용자당 메모리는 최대 10개

class UserInfo(BaseModel):
    name, role, email: str | None
    organization_profile: dict[str, str]   # IdP 디렉토리 정보
    placeholder_values: dict[str, str]     # {{user.<key>}} 치환용
    language: SupportedLanguage | None

class UserMemoryContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: UUID | None
    user_info: UserInfo
    user_preferences: str | None
    memories: tuple[str, ...] = ()

    def without_memories(self) -> "UserMemoryContext":
        """메모리만 비운 복사본 — user_info/preferences는 유지"""

    def as_formatted_list(self) -> list[str]:
        """user_info + preferences + memories를 문자열 리스트 하나로 합침"""
```

`memories`는 Postgres의 `Memory` 테이블에서 가져온 값이고 (`onyx.db.models.Memory`), **사용자당
최대 10개**로 캡이 걸려 있습니다.

`without_memories()`가 따로 있는 이유는 `run_llm_loop`의 `inject_memories_in_prompt` 플래그와
맞물립니다 — **어떤 호출에서는 메모리 주입을 끄고 싶을 때, 사용자 이름/선호 같은 나머지 정보는
그대로 유지한 채 메모리만 뺄 수 있게** 설계돼 있습니다.

## 저장 방식 — 그냥 append가 아니라, "추가/수정"을 LLM이 한 번 더 판단

> 파일: `tools/tool_implementations/memory/memory_tool.py`

```python
class MemoryTool(Tool[MemoryToolOverrideKwargs]):
    NAME = "add_memory"
    # description: "Save memories about the user for future conversations."

    def run(self, placement, override_kwargs, **llm_kwargs):
        memory = llm_kwargs[MEMORY_FIELD]
        memory_text, index_to_replace = process_memory_update(
            new_memory=memory,
            existing_memories=override_kwargs.existing_memories,
            chat_history=override_kwargs.chat_history,
            llm=self.llm,
            user_name=..., user_email=..., user_role=...,
        )
        operation = "update" if index_to_replace is not None else "add"
```

**`add_memory` 도구가 호출되면 그 메모리를 그냥 리스트에 추가하지 않습니다.** `process_memory_update`
(`secondary_llm_flows/memory_update.py`)라는 **별도의 LLM 호출**이 한 번 더 실행돼서, 이 새 메모리가
- 기존 메모리 10개 중 하나와 같은 내용을 갱신하는 것인지(`update`, 어느 인덱스인지까지 판단),
- 아니면 진짜 새로운 정보인지(`add`)

를 판단합니다. 이게 있는 이유를, 코드 주석이 명확히 설명합니다.

> "Not including the Team Information or User Preferences because these are less likely to
> contribute to building the memory. Things like the user's name is important because the LLM
> may create a memory like 'Dave prefers light mode.' instead of 'User prefers light mode.'"

즉 **메모리 문구 자체를 일관된 형태("User prefers...")로 쓰게 만들려고, 도구에 사용자 이름/이메일/
역할까지 같이 넘겨줍니다** — 안 그러면 LLM이 메모리를 3인칭으로 쓰기도 하고 2인칭으로 쓰기도 하는
식으로 일관성이 깨질 수 있어서입니다.

## 정리

```text
저장 트리거: add_memory 도구 호출
판단 로직:   process_memory_update (별도 LLM 호출) → add/update 결정
용량 제한:   사용자당 최대 10개
주입 제어:   UserMemoryContext.without_memories()로 나머지 정보는 유지한 채 메모리만 끌 수 있음
```

