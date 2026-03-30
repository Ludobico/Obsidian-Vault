- [[#Understand-Anything 소개|Understand-Anything 소개]]
- [[#멀티 에이전트 파이프라인 아키텍처|멀티 에이전트 파이프라인 아키텍처]]
- [[#Understand-Anything의 핵심 기능|Understand-Anything의 핵심 기능]]
- [[#Understand-Anything 설치 및 사용법|Understand-Anything 설치 및 사용법]]
- [[#라이선스|라이선스]]


출처 : https://discuss.pytorch.kr/t/understand-anything-claude-code/9418
## Understand-Anything 소개

낯선 코드베이스를 처음 마주할 때 느끼는 막막함은 개발자라면 누구나 공감할 것입니다. 수백 개의 파일이 어떻게 연결되어 있는지, 어떤 함수가 어디서 호출되는지를 파악하기까지 상당한 시간이 걸리며, 특히 팀 온보딩이나 레거시 코드 분석 시에는 생산성이 크게 떨어지기도 합니다. **Understand-Anything**은 이 문제를 해결하기 위해 만들어진 Claude Code 플러그인으로, **어떤 코드베이스든 인터랙티브한 지식 그래프(Knowledge Graph)로 변환하여 파일, 함수, 클래스, 의존성을 한눈에 탐색**할 수 있도록 해줍니다.

![[Pasted image 20260330113115.png]]

Understand-Anything의 핵심 아이디어는 AI가 코드베이스를 분석하여 단순한 파일 목록이 아닌, **각 요소 간의 관계를 이해할 수 있는 구조화된 그래프를 만들어낸다**는 것입니다. 이 플러그인은 Claude Code에 네이티브로 통합되며, Claude Code Marketplace를 통해 한 줄의 명령어로 설치할 수 있습니다. 또한 Codex CLI, OpenClaw, OpenCode, Cursor, Gemini CLI, Pi Agent 등 주요 AI 코딩 도구와도 호환되어 다양한 개발 환경에서 활용할 수 있습니다.

## 멀티 에이전트 파이프라인 아키텍처

Understand-Anything은 내부적으로 여러 전문화된 에이전트(Agent)로 구성된 분석 파이프라인을 통해 동작합니다. 각 에이전트는 독립적인 역할을 담당하며, 이들이 협력하여 코드베이스의 완전한 그래프를 완성합니다.

**project-scanner**는 프로젝트 내 모든 파일을 발견하고 사용된 언어와 프레임워크를 감지합니다.

**file-analyzer**는 각 파일을 분석하여 함수, 클래스, 임포트 정보를 추출하고 그래프의 노드와 엣지를 생성합니다.

**architecture-analyzer**는 API 계층, 서비스 계층, 데이터 계층, UI 계층, 유틸리티 계층 등 아키텍처 레이어를 식별합니다.

**tour-builder**는 의존성 순서에 따라 코드베이스를 학습할 수 있는 가이드 투어를 자동으로 생성하며, **graph-reviewer**는 그래프의 완성도를 검증합니다.

전체 프로젝트는 TypeScript와 React 18, Vite, TailwindCSS v4, React Flow, Zustand, web-tree-sitter, Fuse.js 등의 최신 기술 스택을 기반으로 개발되었습니다.

## Understand-Anything의 핵심 기능

![[Pasted image 20260330113144.png]]

Understand-Anything이 제공하는 인터랙티브 대시보드는 단순한 파일 브라우저를 훨씬 뛰어넘는 기능을 갖추고 있습니다. React Flow를 기반으로 한 **인터랙티브 지식 그래프**는 파일, 함수, 클래스를 클릭 가능한 노드로 시각화하며, 각 노드는 AI가 생성한 자연어 요약을 포함합니다. 이를 통해 코드를 읽지 않고도 각 모듈이 무엇을 하는지 즉시 파악할 수 있습니다.

**가이드 투어(Guided Tours)** 기능은 의존성 순서에 따라 정렬된 아키텍처 워크스루를 자동으로 생성합니다. 주니어 개발자나 새로 합류한 팀원이 코드베이스를 처음부터 체계적으로 이해하는 데 특히 유용합니다. **Diff 임팩트 분석** 기능을 사용하면 특정 코드 변경이 시스템 전체에 어떤 영향을 미치는지를 그래프 상에서 시각적으로 확인할 수 있어, 리팩토링이나 버그 수정 전에 영향 범위를 미리 파악할 수 있습니다.

검색 기능으로는 퍼지(Fuzzy) 매칭과 시맨틱(Semantic) 검색이 모두 지원됩니다. 정확한 이름을 몰라도 의미 기반으로 원하는 코드를 찾을 수 있습니다. **레이어 시각화**는 코드베이스를 아키텍처 레이어별로 자동 그룹화하여 전체 구조를 한눈에 파악할 수 있도록 하며, **페르소나 적응형 UI**는 사용자 유형에 따라 표시되는 세부 정보의 수준을 자동으로 조정합니다.

## Understand-Anything 설치 및 사용법

Understand-Anything을 사용하기 위해서는 먼저 Claude Code를 사용할 수 있는 환경이 준비되어 있어야 합니다.

Claude Code Marketplace를 통한 설치가 가장 간단합니다:

```bash
/plugin marketplace add Lum1104/Understand-Anything
/plugin install understand-anything
```

설치 후 Claude Code 세션에서 다음 명령어로 바로 사용할 수 있습니다:

```bash
/understand                # 코드베이스 전체 분석 실행
/understand-dashboard      # 인터랙티브 대시보드 열기
/understand-chat           # 아키텍처에 대한 질문 대화
/understand-diff           # 코드 변경 영향 분석
/understand-explain        # 특정 파일 심층 분석
/understand-onboard        # 팀 온보딩 가이드 생성
```

로컬에서 직접 개발 환경을 구축하려면 pnpm 패키지 매니저를 사용합니다:

```bash
pnpm install
pnpm --filter @understand-anything/core build
pnpm --filter @understand-anything/skill build
pnpm dev:dashboard   # 대시보드 개발 서버 시작 (포트 기본값)
```

## 라이선스

Understand-Anything 프로젝트는 [MIT 라이선스](https://github.com/Lum1104/Understand-Anything/blob/main/LICENSE?utm_source=pytorchkr&ref=pytorchkr)로 공개되어 있어 개인 및 상업적 목적으로 자유롭게 사용, 수정, 배포할 수 있습니다.

https://github.com/Lum1104/Understand-Anything?utm_source=pytorchkr&ref=pytorchkr

