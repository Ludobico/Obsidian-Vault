
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

> gstack의 진가는 각 명령어(스킬)가 부여받은 **전문 역할(Specialist)** 에 있습니다.

### 제품 기획 및 설계 (Think & Plan)
코드를 작성하기 전, 만들고자 하는 제품이 무엇인지 집요하게 파고들고 기획/설계를 탄탄히 하는 단계입니다.

| 명령어 | 역할 (Specialist) | 하는 일 (주요 기능) |
| :--- | :--- | :--- |
| `/office-hours` | **YC Office Hours** | **(모든 프로젝트의 시작점)** 6가지 질문을 통해 기획을 근본적으로 재구성하고 가설을 검증함. 제품 가치를 10배로 끌어올림. |
| `/plan-ceo-review` | **CEO / Founder** | 단순 기능 구현을 넘어, 사용자 입장에서 마법 같은 '10성급 제품(10-star product)'의 방향성을 모색함. |
| `/plan-eng-review` | **Eng Manager** | 막연한 아이디어를 차단하고 아키텍처, 데이터 흐름, 다이어그램, 엣지 케이스, 테스트 커버리지를 강제로 구체화함. |
| `/plan-design-review` | **Senior Designer** | 인터랙티브 디자인 리뷰 진행. 디자인 차원을 평가(0~10점)하고 만점이 되기 위한 조건을 설명함. |
| `/design-consultation`| **Design Partner** | 처음부터 완전한 디자인 시스템 구축, 창의적 디자인 리스크 제안 및 현실적인 프로덕트 목업 생성. |

### 코드 구현 및 리뷰 (Build & Review)
본격적인 구현과 프로덕션 레벨의 꼼꼼한 코드/디자인 리뷰를 수행합니다.

| 명령어 | 역할 (Specialist) | 하는 일 (주요 기능) |
| :--- | :--- | :--- |
| `/review` | **Staff Engineer** | CI는 통과했지만 실제 프로덕션에서 터질 수 있는 미묘한 버그 탐지, 명백한 버그 자동 수정 및 누락된 구현 지적. |
| `/investigate` | **Debugger** | "조사 없이 수정 없다"는 철칙 아래 데이터 흐름을 추적하고 가설 테스트. (3번 수정 실패 시 시스템 보호를 위해 작업 중단) |
| `/design-review` | **Designer Who Codes**| 라이브 사이트의 시각적 요소 감사 및 직접 수정. 80개 항목 디자인 감사 후 원자적 커밋(Atomic commits)과 스크린샷 제공. |

### QA 및 테스트 (Test)
AI에게 직접 '눈'을 달아주어 브라우저를 조작하고 퀄리티를 보장합니다.

| 명령어 | 역할 (Specialist) | 하는 일 (주요 기능) |
| :--- | :--- | :--- |
| `/qa` | **QA Lead** | 앱을 직접 테스트하고 버그를 찾아 원자적 커밋으로 수정한 뒤 재검증, 회귀 테스트 코드 자동 생성. |
| `/qa-only` | **QA Reporter** | 코드 수정 없이 순수하게 버그 리포트만 생성 (접근 방식은 `/qa`와 동일). |
| `/browse` | **QA Engineer** | **AI에게 '눈'을 달아주는 스킬.** 실제 Chromium 브라우저를 띄워 화면 클릭, 스크린샷 촬영 등 동작 확인. |
| `/setup-browser-cookies`| **Session Manager** | 실제 브라우저(Chrome, Arc 등)에서 쿠키를 추출해 헤드리스(Headless) 세션으로 가져옴. |

### 배포, 보안 감사 및 문서화 (Ship, Secure & Document)
코드를 안전하게 배포하고 시스템을 검증하며 문서를 최신화합니다.

| 명령어 | 역할 (Specialist) | 하는 일 (주요 기능) |
| :--- | :--- | :--- |
| `/ship` | **Release Engineer** | 단일 명령어로 main 브랜치 동기화, 테스트 실행, 커버리지 검토 후 푸시하여 PR 오픈. |
| `/cso` | **Chief Security Officer**| OWASP Top 10 및 STRIDE 기반 보안 감사 (인젝션, 암호화, 접근제어 취약점 스캔). |
| `/document-release` | **Technical Writer** | 방금 배포한 기능에 맞게 프로젝트의 모든 문서 업데이트. 방치된 낡은 README 자동 수정. |

### 멀티 AI 연동 및 안전 장치 (Safety & Utility)
작업 환경을 보호하고, 프로젝트의 건강 상태를 점검합니다.

| 명령어 | 역할 (Specialist) | 하는 일 (주요 기능) |
| :--- | :--- | :--- |
| `/retro` | **Eng Manager** | 팀 맥락을 이해하는 주간 회고 진행 (작업자별 분석, 연속 배포 기록, 테스트 코드 상태 추세 도출). |
| `/codex` | **Second Opinion** | OpenAI Codex CLI를 통한 독립적 교차 검증 제공 (코드 리뷰, 적대적 챌린지 모드). |
| `/careful` | **Safety Guardrails** | `rm -rf`, `DROP TABLE` 등 파괴적인 명령어 실행 전 경고 표시. |
| `/freeze` | **Edit Lock** | 파일 수정 권한을 단일 디렉토리로 제한하여 대형 사고 예방. |
| `/guard` | **Full Safety** | 운영(Prod) 환경 작업을 위해 `/careful`과 `/freeze`를 동시 결합하여 최고 수준의 안전망 가동. |
| `/unfreeze` | **Unlock** | `/freeze`로 걸린 제한 해제. |
| `/gstack-upgrade` | **Self-Updater** | gstack 최신 버전 자동 업그레이드 및 변경 사항 표시. |

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

