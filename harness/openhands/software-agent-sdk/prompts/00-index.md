---
aliases: []
---
- [[#목차|목차]]
- [[#전체 호출 체인|전체 호출 체인]]
- [[#핵심 파일 지도|핵심 파일 지도]]
- [[#두 개의 경로가 공존한다는 점|두 개의 경로가 공존한다는 점]]
- [[#이 폴더가 다루지 않는 것 (다른 프롬프트들)|이 폴더가 다루지 않는 것 (다른 프롬프트들)]]
- [[05-prompt-strings|섹션 본문 원문 (static/dynamic/planning 실제 텍스트 + 번역)]]

# 시스템 프롬프트 조립 구조 개요


> ⚠️ **범위 안내:** 이 폴더는 `AgentBase`가 자신의 시스템 프롬프트를 만드는 메커니즘,
> 즉 **섹션 레지스트리 조립 방식** 하나만 다룹니다. "에이전트 오케스트레이션(도구 호출
> 루프, ReAct 사이클)"과는 다른 주제이고, OpenHands 안에 있는 프롬프트 전체를 다루는
> 것도 아닙니다. 아래 "이 폴더가 다루지 않는 것" 절을 참고해 주세요.

[[openhands]] SDK는 시스템 프롬프트를 하나의 문자열로 만들지 않고, **이름 붙은 섹션(section)들을
정해진 순서로 조립**한 뒤 **static/dynamic 두 블록으로 나눠** 반환합니다. 아래 4개 노트가 이
과정을 단계별로 다룹니다.

## 전체 호출 체인

```text
AgentBase.system_prompt (agent/base.py:202)
  └─ static_system_message (agent/base.py:336-365)
       │
       ├─ [inline system_prompt 있음] → 그대로 반환 (완전 우회)
       │
       ├─ [preset == None] → render_template()          # Jinja escape hatch
       │                        (context/prompts/prompt.py:90)
       │
       └─ [preset in {DEFAULT, PLANNING}]
            └─ create_registry(preset)                   # presets.py:94
                 .build(self._build_prompt_context())     # registry.py:45
                 .static                                  # PromptBlocks.static

AgentBase.dynamic_context (agent/base.py:499-523)
  └─ create_registry(preset or DEFAULT).build(ctx).dynamic
```

## 핵심 파일 지도

| 심볼 | 파일 | 역할 |
|---|---|---|
| `AgentBase.system_prompt` / `static_system_message` | `agent/base.py:202`, `:336` | 진입점, 캐시 가능한 정적 프롬프트 |
| `AgentBase.dynamic_context` | `agent/base.py:499` | 대화별로 변하는 프롬프트 |
| `AgentBase._build_prompt_context` | `agent/base.py:420` | 이번 요청에 필요한 값들을 `PromptContext` 스냅샷으로 고정 |
| `PromptSection`, `CacheTier`, `PromptContext`, `PromptBlocks` | `context/prompts/section.py` | 타입 계약 |
| `PromptRegistry` | `context/prompts/registry.py` | 섹션 조립 엔진 |
| `create_registry`, `PromptPreset` | `context/prompts/presets.py` | 프리셋(모드)별 섹션 구성 |
| `sections/static.py`, `sections/dynamic.py`, `sections/planning.py` | 같은 폴더 | 실제 섹션 구현체 |
| `render_template` | `context/prompts/prompt.py:90` | 레지스트리를 안 쓰는 경우의 Jinja 렌더러(escape hatch) |
## 두 개의 경로가 공존한다는 점

- **레지스트리 경로(기본)** — 파이썬 클래스로 짠 섹션들을 조립합니다. 대부분의 경우가 여기 해당됩니다.
- **Jinja escape hatch** — `system_prompt_filename`을 커스텀 파일로 바꾸거나, `AgentBase`를
  서브클래싱해서 자기만의 `prompt_dir`를 갖는 경우입니다. `render_template()`이 `.j2` 파일을
  직접 렌더링하며, 레지스트리를 아예 거치지 않습니다.

즉 "섹션 조립"은 기본값일 뿐이고, 완전히 다른 프롬프트가 필요하면 Jinja로 통째로 갈아끼울 수
있게 열어둔 구조입니다.

## 이 폴더가 다루지 않는 것 (다른 프롬프트들)

`AgentBase`의 시스템 프롬프트 말고도, OpenHands 안에는 이 레지스트리와 무관한 프롬프트
템플릿이 따로 존재합니다.

- `context/prompts/templates/ask_agent_template.j2` — 실행 도중 사용자가 던진 질문을
  "도구 호출 없이 답만 하라"고 감싸는 템플릿입니다. 시스템 프롬프트가 아니라 질문
  메시지용입니다. 원문/번역은 [[05-prompt-strings]] 참고.
- `context/prompts/templates/skill_knowledge_info.j2` — 키워드 매칭으로 트리거된 스킬의
  지식을 대화 중간에 주입하는 독립 Jinja 템플릿입니다. 섹션 레지스트리와 별개로 동작합니다.
  원문/번역은 [[05-prompt-strings]] 참고.
- 이 밖에도 서브에이전트 프롬프트, condenser(대화 압축) 프롬프트, 보안 정책 파일 등이
  코드베이스 다른 위치에 있는데, 아직 분석하지 않았습니다.

> static/dynamic/planning 섹션들이 실제로 담고 있는 프롬프트 텍스트 전문(원문 + 한국어 번역)은
> [[05-prompt-strings]]에 정리했습니다.

이 폴더는 "OpenHands 프롬프트 전체 지도"가 아니라 **"에이전트 시스템 프롬프트 하나를
어떻게 조립하는가"** 에 한정된 자료입니다.

