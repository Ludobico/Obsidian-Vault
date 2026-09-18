# Onyx 개요

#harness #onyx #overview

**사내 지식 소스(Jira, Confluence, Slack, Notion, Google Drive, Salesforce 등 60개 이상)를 연결해서,
"우리 회사 전용 ChatGPT"를 만들어주는 오픈소스 플랫폼입니다.** 원래 이름은 Danswer였고, 이후 Onyx로
리브랜딩되었습니다 (저작권 표시는 지금도 `DanswerAI, Inc.`로 남아있습니다).

## 규모
`gitnexus` 인덱스 기준입니다.

```text
파일 수:        7,142
심볼 수:        110,412
관계(엣지) 수:   256,245
기능 모듈 수:    100개 이상 (cohesion 66~94%)
```

상위 모듈만 봐도 이렇습니다.

```text
Db(1402)  Components(1334, 프론트엔드)  Chat(531)  Cmd(500)  Auth(344)
Scim(283)  Indexing(262)  Llm(241)  Craft(231)  Outlook(225)
Install(213)  Zoom(192)  Sandbox_proxy(179)  Opensearch(164)  Mcp(155)
Slack(147)  Skills(146)  Sandbox(133)  Gateway(133)  Hooks(114) ...
```

`Craft`, `Outlook`, `Zoom`, `Slack` 등이 각각 100개 이상의 심볼을 가진 **독립 모듈**이라는 게
핵심입니다 — Onyx는 "RAG 엔진 하나"가 아니라 **"수십 개의 사내 시스템 연동을 각각 하나의
서브시스템으로 다루는 플랫폼"** 입니다.

## 아키텍처 형태

Onyx는 코드로 임베드하는 SDK가 아니라, 통째로 배포하는 **제품(서비스)** 입니다.

```text
스택:   FastAPI 백엔드(Python) + Next.js 프론트엔드(web/)
        + Postgres(메타데이터: 사용자, Persona, 대화 기록)
        + Vespa(벡터/키워드 문서 인덱스)
        + Redis
        + 백그라운드 워커(커넥터 동기화)

재사용 방식: 내부 로직을 import하는 게 아니라, REST API(/query, /chat)로 접근

배포 단위: 전체 스택을 Docker/K8s로 배포하는 하나의 서비스
```

## 배포 모드 두 가지

```text
Lite     — 최소 Chat UI, 1GB 미만 메모리
Standard — 벡터/키워드 인덱싱 + 백그라운드 워커 + AI 추론 서버 + Redis/MinIO 풀스택
```

## 엔터프라이즈/멀티테넌트 기능

```text
SSO         — Google OAuth, OIDC, SAML
SCIM        — 사용자 프로비저닝/동기화
RBAC        — 에이전트·액션 단위 역할 기반 접근 제어
멀티테넌트   — tenant_id로 테넌트별 데이터 격리
```

## 라이선스 구조 — MIT + 예외 구역

```text
전체:  MIT License (Copyright DanswerAI, Inc.)
예외:  backend/ee/, web/src/app/ee/, web/src/ee/  → Onyx Enterprise License
```

`ee/` 밑 디렉토리는 별도 라이선스(Onyx Enterprise License)가 적용됩니다. 예를 들어 Jira
프로젝트/이슈 단위 권한(permission)을 실제로 조회하는 로직이 이 경계 안에 게이팅되어 있습니다.


