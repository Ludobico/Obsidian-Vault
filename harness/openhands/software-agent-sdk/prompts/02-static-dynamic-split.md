# static / dynamic 2블록 분리

## 조립 엔진 (`PromptRegistry.build`)

```python
def build(self, ctx: PromptContext) -> PromptBlocks:
    buckets: defaultdict[CacheTier, list[str]] = defaultdict(list)
    for section in self._sections.values():          # 등록 순서 그대로 순회
        if not section.guard(ctx):
            continue
        text = section.render(ctx)
        if text is None or not text.strip():
            continue
        buckets[section.cache_tier].append(text.strip())

    return PromptBlocks(
        static="\n\n".join(buckets[CacheTier.STATIC]),
        dynamic="\n\n".join(buckets[CacheTier.DYNAMIC]) or None,
    )
```

동작을 도식화하면 다음과 같습니다.

```text
등록 순서대로 순회
┌─────────────┐   guard=False    ┌──────────┐
│ Section A   │ ───────────────▶ │  스킵됨   │
└─────────────┘                  └──────────┘

┌─────────────┐   guard=True     ┌──────────────┐   cache_tier=STATIC
│ Section B   │ ───────────────▶ │ render() 호출 │ ───────────────────▶ static 버킷에 append
└─────────────┘                  └──────────────┘

┌─────────────┐   guard=True     ┌──────────────┐   cache_tier=DYNAMIC
│ Section C   │ ───────────────▶ │ render() 호출 │ ───────────────────▶ dynamic 버킷에 append
└─────────────┘                  └──────────────┘

마지막에:
  static  = "\n\n".join(static 버킷)
  dynamic = "\n\n".join(dynamic 버킷)  or  None
```

## 왜 나누는가 — 프롬프트 캐싱

- `static_system_message` (`agent/base.py:336`) → **대화 간 재사용 가능한 시스템 프롬프트**입니다.
  정체성, 규칙, 툴 설명처럼 거의 안 바뀌는 내용만 여기 들어갑니다.
- `dynamic_context` (`agent/base.py:499`) → **대화마다 새로 계산해야 하는 부분**입니다 (현재 시각,
  레포 스킬, 시크릿 목록 등). 캐시 마커 없이 별도 content block으로 전송됩니다.

이렇게 나누는 이유는 docstring에 명시돼 있습니다:

> "This content should NOT be included in the cached system prompt to enable
> cross-conversation cache sharing. Instead, it is sent as a second content
> block (without a cache marker) inside the system message."

즉 **static 블록은 캐시 히트를 노리고, dynamic 블록만 매번 다시 계산**해서 전체 프롬프트를
매번 새로 캐싱하는 비용을 피합니다.

## 순서 설계 — 가장 변동성 큰 값은 맨 뒤로

> 파일: `context/prompts/presets.py:82-91`

```python
_DYNAMIC_SECTIONS: Final[tuple[PromptSection, ...]] = (
    RepoContextSection(),
    MemoryContextSection(),
    AvailableSkillsSection(),
    CustomSuffixSection(),
    CustomSecretsSection(),
    # DateTimeSection is intentionally last: it is the only per-conversation
    # volatile value, so the stable dynamic content stays a cache-friendly prefix.
    DateTimeSection(),
)
```

```text
dynamic 블록 내부 순서 (캐시 관점에서 본 안정성):

[REPO_CONTEXT] [MEMORY_CONTEXT] [SKILLS] [CUSTOM_SUFFIX] [CUSTOM_SECRETS] [DATETIME]
└──────────────── 상대적으로 안정적 (prefix) ─────────────┘   └─ 매번 바뀜 (suffix)
```

같은 대화 안에서 레포/스킬/시크릿 목록은 안 바뀌는데 시각만 바뀝니다. **변하는 값을
맨 끝에 둬야, 그 앞부분(prefix)까지는 여전히 캐시 히트가 납니다.** 변하는 값을 앞에
두면 그 뒤 전체가 매번 캐시 미스로 처리됩니다.

