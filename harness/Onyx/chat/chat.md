#harness #agent-loop

# Chat 도메인 개요 — Onyx의 에이전트 하네스

**Onyx의 "에이전트 하네스"는 별도 프레임워크가 아니라 `backend/onyx/chat/` 밑의 함수 몇 개로 이루어진
하나의 파이프라인입니다.** [[Onyx]] 전체 아키텍처 중 실제로 "LLM이 도구를 반복 호출하며 답을
만들어가는" 부분이 전부 이 폴더에 있습니다.

## 개념 — Turn과 Cycle

> 파일: `chat/README.md`

```text
Turn  — 사용자가 메시지 1개를 보내고 에이전트가 최종 답변을 낼 때까지의 전체 처리
Cycle — 그 안에서 LLM 추론 1회 (도구 호출 여부와 무관)
```

## 파이프라인 — 요청 1개가 흘러가는 순서

```text
1. process_message._run_models()      최상위 진입점 — 검증/셋업, 모델별 워커 스레드 기동
2.   └─ run_llm_loop()                워커 스레드 안에서 도는 진짜 하네스 루프 (Turn 전체 담당)
3.       ├─ construct_message_history()   사이클마다 히스토리를 토큰 예산 안으로 truncate
4.       ├─ build_system_prompt()         사이클마다 시스템 프롬프트 재조립
5.       ├─ run_llm_step()                LLM 추론 1회, 스트리밍 델타를 패킷으로 변환
6.       └─ run_tool_calls()              뽑힌 tool_calls를 병렬 실행, 결과를 히스토리에 반영
7.   Emitter → merged_queue → Writer → tee → Reader   모든 단계의 패킷이 이 경로로 클라이언트까지 감
```

## 이 폴더의 노트

| 노트 | 다루는 함수/개념 |
|---|---|
| [[agent-loop]] | `run_llm_loop` — while문 자체, 사이클 구조, tool_choice 전환, 종료조건 (하네스의 심장) |
| [[llm-step]] | `run_llm_step` — LLM 스트림 델타를 reasoning/answer/tool_call로 분류 |
| [[context-window]] | `construct_message_history` + `token_budget.py` — 히스토리 truncation, 토큰 예산 |
| [[prompt-assembly]] | `build_system_prompt` — 시스템 프롬프트 조립 순서 |
| [[prompt-strings]] | 조립되는 문자열 상수들의 실제 원문 + 한국어 번역 |
| [[tool-execution]] | `run_tool_calls` — 도구 병합, 병렬 실행, 실패 격리 |
| [[streaming-pipeline]] | `_run_models` — 워커/Writer/Reader 3단 스레드 구조, 취소 처리, 저장 보장 |
| [[memory]] | `UserMemoryContext`, `add_memory` — 대화 중 학습되는 사용자 메모리 |

## 3중 메시지 표현 — 왜 이렇게 나뉘어 있는가

> 파일: `chat/README.md`, `chat/models.py`

```text
ChatMessage        DB 모델. 로드 직후 ChatMessageSimple로 변환하고 그 이후로는 안 씀
ChatMessageSimple  코드베이스 전체가 쓰는 canonical 표현 — 메시지 구조를 바꾸려면 여기부터
LanguageModelInput LLM에 실제로 보내는 최소 표현 — translate_history_to_llm_format()이 변환
```

[[context-window]]가 다루는 `construct_message_history`는 `ChatMessageSimple` 리스트를 받고
반환하며, [[llm-step]]의 `translate_history_to_llm_format`이 그걸 마지막 순간에
`LanguageModelInput`(OpenAI 스타일 메시지)으로 변환합니다 — **하네스 내부 로직은 전부
`ChatMessageSimple` 기준으로 짜여 있고, provider별 메시지 포맷 차이는 이 변환 지점 하나로
격리**돼 있습니다.

## 하네스 밖의 것들 — 이 위에 얹힌 인스턴스들

`chat/` 폴더가 하네스 본체이고, 아래는 전부 이 하네스를 재사용/변형해서 만든 별도 기능입니다
(같은 vault에 추후 별도 폴더로 정리 예정).

```text
coding_agent/, tools/fake_tools/coding_agent.py   — bash_tool + code-interpreter로 도는 서브 에이전트
deep_research/dr_loop.py (run_deep_research_llm_loop) — run_llm_loop과 별도인 멀티스텝 리서치 루프
tools/tool_implementations/*                       — 하네스가 호출하는 개별 도구 구현체
```

## 관련 노트

- [[Onyx]] — 프로젝트 전체 개요
- [[retrieval]] — 검색/ACL 도메인 (하네스가 `SearchTool`을 통해 호출하는 쪽)
