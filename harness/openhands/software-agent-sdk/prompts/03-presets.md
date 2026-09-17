# 프리셋 (`PromptPreset`)

> 파일: `context/prompts/presets.py`

```python
class PromptPreset(StrEnum):
    DEFAULT = "default"
    PLANNING = "planning"
```

"모드 전환"을 if/else 분기가 아니라 **섹션 리스트 자체를 통째로 교체**하는 방식으로 구현했습니다.

## `DEFAULT` — 18개 static 섹션 + 6개 공유 dynamic 섹션

```python
_DEFAULT_STATIC_SECTIONS = (
    SoulSection(), RoleSection(), MemorySection(), EfficiencySection(),
    FileSystemSection(), CodeQualitySection(), VersionControlSection(),
    PullRequestsSection(), ProblemSolvingSection(), SelfDocumentationSection(),
    SecuritySection(),              # guard: security_policy_filename set
    SecurityRiskAssessmentSection(), # guard: llm_security_analyzer
    BrowserSection(),                # guard: ctx.enable_browser
    ExternalServicesSection(), EnvironmentSetupSection(),
    TroubleshootingSection(), ProcessManagementSection(),
    ModelSpecificSection(),          # guard: model_family resolved
)
```

## `PLANNING` — 완전히 다른 구성인데, 섹션은 하나뿐

```python
_PLANNING_STATIC_SECTIONS = (PlanningSection(),)
```

여기가 흥미로운 지점입니다. `PlanningSection` (`sections/planning.py:21-134`)은 **레지스트리
철학(잘게 쪼갠 가드 섹션들)을 따르지 않습니다** — 자체 docstring이 이렇게 설명합니다.

> "Unlike the default composition — whose blocks each carry a guard and can be
> overridden individually — the planning prompt is a single standalone STATIC
> block with no per-section guards, so it is one section."

즉 `<ROLE>`, `<IMPORTANT_PRINCIPLES>`, `<EFFICIENCY>`, `<PLANNING_WORKFLOW>`(Phase 1~4),
`<PLAN_SCOPE>`, `<PLAN_STRUCTURE>`가 전부 파이썬 문자열 하나 안에 통짜로 들어있고, 유일한
치환값은 `plan_structure` 하나뿐입니다.

```python
def render(self, ctx: PromptContext) -> str | None:
    plan_structure = str(ctx.template_kwargs.get("plan_structure", ""))
    return self._BODY.replace("{plan_structure}", plan_structure)
```

**교훈:** 섹션 레지스트리가 "무조건 잘게 쪼개라"를 강제하지 않습니다. 재사용/재조합 가치가
없는 프롬프트(플래닝 모드는 다른 모드와 섞일 일이 없음)는 통짜 섹션 하나로 남겨둬도 됩니다.
어디까지 쪼갤지는 "이 조각이 다른 프리셋과 섞일 가능성이 있는가"로 판단한 것으로 보입니다.

## 두 프리셋이 공유하는 것 — dynamic 섹션

```python
def create_registry(preset: PromptPreset = PromptPreset.DEFAULT) -> PromptRegistry:
    match preset:
        case PromptPreset.PLANNING:
            static_sections = _PLANNING_STATIC_SECTIONS
        case PromptPreset.DEFAULT:
            static_sections = _DEFAULT_STATIC_SECTIONS

    r = PromptRegistry()
    for section in (*static_sections, *_DYNAMIC_SECTIONS):   # ← dynamic은 항상 동일
        r.register(section)
    return r
```

```text
              ┌─────────────────────┐        ┌─────────────────────┐
              │ DEFAULT static (18) │        │ PLANNING static (1) │
              └─────────────────────┘        └─────────────────────┘
                         │                              │
                         └──────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │  DYNAMIC 섹션 6개 (항상 동일)   │
                        │  Repo/Memory/Skills/Suffix/     │
                        │  Secrets/DateTime               │
                        └───────────────────────────────┘
```

static 섹션 목록만 바뀌고 dynamic 섹션은 프리셋과 무관하게 항상 붙습니다 — "플래닝 모드로
전환해도 레포 컨텍스트나 시크릿 정보는 여전히 필요하다"는 판단으로 보입니다.

## Escape hatch — 레지스트리를 아예 안 쓰는 경우

> 파일: `context/prompts/prompt.py`, `agent/base.py:352-365`

```python
if self.system_prompt is not None:
    return self.system_prompt              # ① 완전 우회: 문자열 그대로 사용

preset = self._prompt_preset                # _PRESET_BY_FILENAME 조회, 못 찾으면 None
if preset is None:
    return render_template(                 # ② Jinja escape hatch
        prompt_dir=self.prompt_dir,
        template_name=self.system_prompt_filename,
        **self._resolved_template_kwargs(),
    )

return create_registry(preset).build(self._build_prompt_context()).static  # ③ 레지스트리 경로
```

`②`로 빠지는 경우는 두 가지입니다: `system_prompt_filename`을 알려진 프리셋 이름이 아닌
커스텀 파일로 바꿨거나, `AgentBase`를 서브클래싱해서 자기만의 `prompt_dir`를 갖는 경우입니다.

`render_template` (`prompt.py:90-116`)은 순수 Jinja2 렌더러입니다.

```python
def render_template(prompt_dir: str, template_name: str, **ctx) -> str:
    tpl = _get_template(prompt_dir, template_name)   # FileSystemBytecodeCache로 캐시
    return refine(tpl.render(**ctx).strip())
```

- `FlexibleFileSystemLoader` — 상대 경로(`prompt_dir` 기준)와 절대 경로를 모두 지원합니다.
- `FileSystemBytecodeCache` — 프로세스 간에도 템플릿 재파싱을 피합니다.
- `refine()` — Windows에서 `terminal`→`execute_powershell`, `bash`→`powershell` 치환을
  수행합니다 (플랫폼별 툴 이름 불일치를 렌더링 이후 정규식으로 보정합니다). 레지스트리 경로의
  `sections/static.py:_refine`과 동일한 로직입니다 — Jinja 경로도 static 섹션 경로도 이
  보정을 거칩니다.

**요약:** 레지스트리는 기본 경로일 뿐이고, 완전히 다른 프롬프트 시스템(순수 Jinja)이
같은 진입점(`static_system_message`) 뒤에 숨어서 공존합니다.

