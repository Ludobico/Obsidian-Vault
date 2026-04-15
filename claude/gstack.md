
- [[#gstack 소개 - **제작자**: Garry Tan (Y Combinator CEO)|gstack 소개 - **제작자**: Garry Tan (Y Combinator CEO)]]
- [[#gstack 핵심 명령어 (Commands)|gstack 핵심 명령어 (Commands)]]
	- [[#gstack 핵심 명령어 (Commands)#제품 기획 및 설계 (Think & Plan)|제품 기획 및 설계 (Think & Plan)]]
	- [[#gstack 핵심 명령어 (Commands)#코드 구현 및 리뷰 (Build & Review)|코드 구현 및 리뷰 (Build & Review)]]
	- [[#gstack 핵심 명령어 (Commands)#QA 및 테스트 (Test)|QA 및 테스트 (Test)]]
	- [[#gstack 핵심 명령어 (Commands)#배포, 보안 감사 및 문서화 (Ship, Secure & Document)|배포, 보안 감사 및 문서화 (Ship, Secure & Document)]]
	- [[#gstack 핵심 명령어 (Commands)#멀티 AI 연동 및 안전 장치 (Safety & Utility)|멀티 AI 연동 및 안전 장치 (Safety & Utility)]]
- [[#설치 및 사용 워크플로우|설치 및 사용 워크플로우]]
	- [[#설치 및 사용 워크플로우#요구 사항|요구 사항]]
	- [[#설치 및 사용 워크플로우#글로벌 설치 (터미널)|글로벌 설치 (터미널)]]

# gstack: Y Combinator 대표 Garry Tan의 Claude Code 스킬 모음

## gstack 소개 - **제작자**: Garry Tan (Y Combinator CEO)

- **개요**: Claude Code를 **20인 규모의 가상 엔지니어링 팀**처럼 활용할 수 있게 해주는 오픈소스 소프트웨어 팩토리(스킬 모음)입니다. 
- **특징**: AI를 단순한 코드 생성 도구로 쓰지 않고 CEO, 디자이너, 엔지니어링 매니저, QA 리드 등 **각 역할에 특화된 구조화된 워크플로우**를 부여합니다. 
- **기본 워크플로우**: `Think` → `Plan` → `Build` → `Review` → `Test` → `Ship` → `Reflect`

## gstack 핵심 명령어 (Commands)

> gstack의 진정한 가치는 단순한 명령어 모음이 아니라, 각 명령어(스킬)에 부여된 **전문적인 역할(Specialist)** 에 있습니다. 마치 하나의 개발 팀을 운영하듯 상황에 맞는 전문가를 호출하여 작업을 진행할 수 있습니다.

### 제품 기획 및 설계 (Think & Plan)
코드를 섣불리 작성하기 전에, 우리가 진정으로 만들고자 하는 제품이 무엇인지 깊이 고민하고 설계의 뼈대를 튼튼하게 다지는 단계입니다.

| 명령어                    | 역할 (Specialist)     | 하는 일 (주요 기능)                                                                                               |
| :--------------------- | :------------------ | :--------------------------------------------------------------------------------------------------------- |
| `/office-hours`        | **YC Office Hours** | **(프로젝트의 첫 단추)** 6가지 핵심 질문을 던져 기획의 근본적인 방향을 재구성하고 가설을 검증합니다. 제품이 지닌 잠재 가치를 극대화하는 데 집중합니다.                  |
| `/plan-ceo-review`     | **CEO / Founder**   | 단순히 기능을 구현하는 것을 넘어, 실제 사용자 입장에서 감동을 줄 수 있는 '10성급(10-star)' 제품의 비전과 방향성을 함께 모색합니다.                          |
| `/plan-eng-review`     | **Eng Manager**     | 모호한 아이디어 상태로 개발에 착수하는 것을 방지합니다. 시스템 아키텍처, 데이터 흐름, 예외 처리(Edge case), 테스트 커버리지 등을 강제적으로 구체화하여 탄탄한 설계를 유도합니다. |
| `/plan-design-review`  | **Senior Designer** | 대화형 디자인 리뷰를 진행합니다. 현재 구상 중인 디자인의 수준을 0점부터 10점 사이로 평가하고, 만점을 받기 위해 보완해야 할 구체적인 조건들을 제시해 줍니다.                |
| `/design-consultation` | **Design Partner**  | 프로젝트 초기 단계부터 일관된 디자인 시스템을 구축하도록 돕습니다. 창의적이면서도 과감한 디자인 방향성을 제안하며, 현실성 있는 프로덕트 목업(Mockup)을 생성해 줍니다.          |

### 코드 구현 및 리뷰 (Build & Review)
본격적인 구현과 프로덕션 레벨의 꼼꼼한 코드/디자인 리뷰를 수행합니다.

| 명령어              | 역할 (Specialist)        | 하는 일 (주요 기능)                                                                                                                     |
| :--------------- | :--------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| `/review`        | **Staff Engineer**     | 단순한 CI 통과 여부를 넘어, 실제 프로덕션 환경에서 발생할 수 있는 치명적이고 미묘한 버그를 찾아냅니다. 명백한 오류는 자동으로 수정하며, 누락된 로직이나 개선점을 예리하게 지적합니다.                        |
| `/investigate`   | **Debugger**           | "충분한 조사 없이 함부로 코드를 수정하지 않는다"는 원칙을 따릅니다. 데이터 흐름을 끝까지 추적하고 가설을 검증하며 디버깅을 수행합니다. (안전망: 3번 이상 수정에 실패하면 시스템 보호를 위해 스스로 작업을 중단합니다.)    |
| `/design-review` | **Designer Who Codes** | 현재 구현된 화면의 시각적인 요소들을 감사(Audit)하고 직접 코드를 수정합니다. 80여 개의 디자인 체크리스트를 기반으로 점검한 뒤, 기능 단위의 깔끔한 커밋(Atomic commits)과 결과 스크린샷을 함께 제공해 줍니다. |

### QA 및 테스트 (Test)
AI에게 화면을 볼 수 있는 '시각적 인지 능력(눈)'을 달아주어, 실제 사용자와 같은 환경에서 브라우저를 조작하며 소프트웨어의 품질을 보장합니다.

| 명령어                      | 역할 (Specialist)     | 하는 일 (주요 기능)                                                                                                                     |
| :----------------------- | :------------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| `/qa`                    | **QA Lead**         | 애플리케이션을 직접 구동하고 테스트하여 버그를 찾아냅니다. 발견된 버그를 커밋 단위로 깔끔하게 수정한 뒤 다시 검증하며, 이후 동일한 문제가 발생하지 않도록 회귀 테스트(Regression test) 코드까지 자동으로 작성합니다. |
| `/qa-only`               | **QA Reporter**     | 접근 방식은 `/qa`와 동일하지만, 코드를 직접 수정하지 않고 순수하게 발견된 버그에 대한 상세 리포트만 생성해 줍니다.                                                             |
| `/browse`                | **QA Engineer**     | **AI에게 직접 눈을 달아주는 핵심 스킬입니다.** 실제 크로미움(Chromium) 브라우저를 띄워 화면을 클릭하거나 스크린샷을 촬영하는 등, 브라우저 상의 동작을 직접 눈으로 확인하며 점검합니다.                  |
| `/setup-browser-cookies` | **Session Manager** | 사용 중인 실제 브라우저(Chrome, Arc 등)의 쿠키 및 세션 정보를 추출하여, AI가 사용하는 헤드리스(Headless) 브라우저 환경으로 안전하게 연동해 줍니다.                                  |

### 배포, 보안 감사 및 문서화 (Ship, Secure & Document)
완성된 코드를 안전하게 배포하고, 보안 취약점을 검증하며, 프로젝트의 문서를 항상 최신 상태로 유지하도록 돕습니다.

| 명령어                 | 역할 (Specialist)            | 하는 일 (주요 기능)                                                                                                      |
| :------------------ | :------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| `/ship`             | **Release Engineer**       | 명령어 하나로 메인(main) 브랜치와 동기화하고, 전체 테스트를 실행하며 커버리지를 검토합니다. 모든 검증이 끝나면 코드를 푸시(Push)하고 자동으로 PR(Pull Request)까지 생성해 줍니다. |
| `/cso`              | **Chief Security Officer** | OWASP Top 10 및 STRIDE 위협 모델링을 기반으로 프로젝트 전반의 보안 감사를 수행합니다. 인젝션, 암호화 결함, 접근 제어 등 치명적인 취약점을 스캔하고 대비책을 마련합니다.         |
| `/document-release` | **Technical Writer**       | 새롭게 배포된 기능과 변경 사항에 맞추어 프로젝트 내 관련된 모든 문서를 일괄적으로 업데이트합니다. 오랫동안 방치되어 낡은 내용이 담긴 README 파일 등도 최신 상태로 다듬어 줍니다.          |

### 멀티 AI 연동 및 안전 장치 (Safety & Utility)
개발 환경을 안전하게 보호하며, 프로젝트와 팀의 전반적인 건강 상태를 점검하고 관리합니다.

| 명령어               | 역할 (Specialist)       | 하는 일 (주요 기능)                                                                                  |
| :---------------- | :-------------------- | :-------------------------------------------------------------------------------------------- |
| `/retro`          | **Eng Manager**       | 팀의 작업 맥락을 파악하여 주간 회고를 진행합니다. 작업자별 기여도 분석, 연속 배포 기록, 테스트 코드의 건강 상태 추세 등을 도출하여 프로젝트의 흐름을 짚어줍니다. |
| `/codex`          | **Second Opinion**    | 다른 AI 모델(OpenAI Codex)을 활용하여 현재 작성된 로직을 독립적으로 교차 검증받습니다. 객관적인 코드 리뷰나 비판적인 검토가 필요할 때 유용합니다     |
| `/careful`        | **Safety Guardrails** | `rm -rf`나 `DROP TABLE`처럼 시스템에 치명적인 영향을 줄 수 있는 파괴적인 명령어가 실행되기 전에 경고를 표시하여 실수를 방지합니다.           |
| `/freeze`         | **Edit Lock**         | AI가 코드를 수정할 수 있는 범위를 특정 디렉토리 하나로 완전히 제한합니다. 의도치 않은 파일이 덮어씌워지거나 삭제되는 대형 사고를 예방할 수 있습니다.        |
| `/guard`          | **Full Safety**       | 실제 운영(Production) 환경에서 작업할 때, `/careful`과 `/freeze` 기능을 동시에 가동하여 프로젝트를 가장 강력하게 보호합니다.         |
| `/unfreeze`       | **Unlock**            | `/freeze` 명령어로 설정된 디렉토리 수정 제한을 해제하여 다시 자유롭게 작업할 수 있도록 만듭니다.                                   |
| `/gstack-upgrade` | **Self-Updater**      | 명령어 하나로 gstack을 최신 버전으로 안전하게 업그레이드하며, 새롭게 추가되거나 변경된 사항들을 직관적으로 보여줍니다.                         |

---

## 설치 및 사용 워크플로우

### 요구 사항
- Claude Code, Git, Bun v1.0 이상 (Windows 환경의 경우 Node.js 필수)

### 글로벌 설치 (터미널)

https://github.com/garrytan/gstack?utm_source=pytorchkr&ref=pytorchkr

```bash
git clone --single-branch --depth 1 [https://github.com/garrytan/gstack.git](https://github.com/garrytan/gstack.git) ~/.claude/skills/gstack
cd ~/.claude/skills/gstack
./setup
```


## example

```
You:    I want to build a daily briefing app for my calendar.
You:    /office-hours
Claude: [asks about the pain — specific examples, not hypotheticals]

You:    Multiple Google calendars, events with stale info, wrong locations.
        Prep takes forever and the results aren't good enough...

Claude: I'm going to push back on the framing. You said "daily briefing
        app." But what you actually described is a personal chief of
        staff AI.
        [extracts 5 capabilities you didn't realize you were describing]
        [challenges 4 premises — you agree, disagree, or adjust]
        [generates 3 implementation approaches with effort estimates]
        RECOMMENDATION: Ship the narrowest wedge tomorrow, learn from
        real usage. The full vision is a 3-month project — start with
        the daily briefing that actually works.
        [writes design doc → feeds into downstream skills automatically]

You:    /plan-ceo-review
        [reads the design doc, challenges scope, runs 10-section review]

You:    /plan-eng-review
        [ASCII diagrams for data flow, state machines, error paths]
        [test matrix, failure modes, security concerns]

You:    Approve plan. Exit plan mode.
        [writes 2,400 lines across 11 files. ~8 minutes.]

You:    /review
        [AUTO-FIXED] 2 issues. [ASK] Race condition → you approve fix.

You:    /qa https://staging.myapp.com
        [opens real browser, clicks through flows, finds and fixes a bug]

You:    /ship
        Tests: 42 → 51 (+9 new). PR: github.com/you/app/pull/42
```

