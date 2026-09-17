# 섹션 프로토콜 (`PromptSection`)

> 파일: `openhands-sdk/openhands/sdk/context/prompts/section.py`

프롬프트의 최소 단위입니다. "하나의 가드 조건 + 하나의 텍스트 블록"으로 정의됩니다.

```python
class PromptSection(Protocol):
    name: str
    cache_tier: CacheTier

    def guard(self, ctx: PromptContext) -> bool: ...
    def render(self, ctx: PromptContext) -> str | None: ...
```

## `guard()` vs `render()` — 왜 나눴는가

두 메서드는 서로 다른 질문에 답합니다.

```text
guard(ctx) == False
  → "이 섹션은 이 상황에 아예 해당 안 됨" (예: 브라우저 비활성화, 플랫폼 안 맞음)
  → render()는 호출조차 되지 않음

guard(ctx) == True  &&  render(ctx) is None (or 빈 문자열)
  → "적용은 되는 상황인데, 지금 낼 내용이 없음" (예: 스킬이 하나도 없음)
```

이 구분 덕분에 "조건부 존재"와 "조건부 내용"을 섞어서 하나의 if문에 욱여넣지 않습니다.
각 섹션은 이 두 메서드만 순수하게(I/O 없이) 구현하면 되므로 개별 유닛테스트가 쉽습니다.

## `PromptContext` — 렌더링에 필요한 모든 것의 불변 스냅샷

```python
class PromptContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    template_kwargs: Mapping[str, object]      # soul_content, enable_browser, model_name...
    tool_names: tuple[str, ...]
    platform: Platform                          # WINDOWS / MACOS / LINUX / OTHER
    working_dir: str | None
    now: str | None                             # 포맷된 현재 시각
    skill_names: tuple[str, ...]
    secret_names: tuple[str, ...]
    repo_skills: tuple[tuple[str, str], ...]     # (이름, 내용) 쌍
    available_skills_prompt: str | None
    custom_suffix: str | None
    memory_context: str | None
    secret_infos: tuple[tuple[str, str | None], ...]
```

- `frozen=True` + `template_kwargs`를 `MappingProxyType`으로 감싸서 **렌더링 도중 값이
  바뀌는 것을 원천 차단**합니다 (섹션이 실수로 컨텍스트를 변형하는 사고를 방지합니다).
- `enable_browser`, `model_family`, `cli_mode`는 `template_kwargs`를 감싼 타입 있는
  `@property`입니다 — 원시 dict 접근 대신 섹션 코드가 타입 안전하게 조회하도록 해줍니다.

## `CacheTier` — 섹션이 어느 블록으로 갈지

```python
class CacheTier(StrEnum):
    STATIC = "static"    # 대화 간 캐시 재사용 가능
    DYNAMIC = "dynamic"  # 대화(턴)마다 새로 계산
```

이 값 하나로 한 섹션의 출력이 정적/동적 블록 중 어디로 모일지 결정됩니다 (자세한 내용은
[[02-static-dynamic-split]] 참고).

## `PromptBlocks` — 최종 산출물

```python
class PromptBlocks(NamedTuple):
    static: str
    dynamic: str | None = None
```

`SystemPromptEvent`가 갖는 두 개의 content block에 1:1로 대응합니다.

