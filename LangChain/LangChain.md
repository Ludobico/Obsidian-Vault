랭체인(LangChain)은 해리슨 페이스에 의해 2022년 10월 오픈 소스 프로젝트로 시작되었습니다. 그는 머신러닝 스타트업인 로버스트 인텔리전스(Robust intelligence)에서 근무하면서 LLM을 활용하여 애플리케이션과 파이프라인을 신속하게 구축할 수 있는 플랫폼의 필요성을 느꼈습니다. 이러한 비전을 가지고 개발자들이 챗봇, 질의응답 시스템, 자동 요약 등 다양한 LLM 애플리케이션을 쉽게 개발할 수 있도록 지원하는 프레임워크를 만들었습니다.

## LangChain V0.1.0
---
랭체인 v0.1.0 버전은 2024년 1월에 출시되었습니다. 랭체인 프로젝트의 stable 버전으로, 완전한 하위 호환성을 제공하는 것을 목표로 공개되었습니다.

랭체인은 stable 버전을 출시하면서, 두 가지 주요 아키텍처를 변경했습니다. `langchain-core`를 별도로 분리하여 추상화, 인터페이스, 핵심 기능을 `langchain-core`에 포함시켰습니다.

또한 `langchain` 에서 파트너 패키지를 분리하여 `langchain-community` 와 독립적인 파트너 패키지(`langchain-openai` 등)을 구분하여 제공합니다.

## Installation
---

```bash
pip install langchain
```

```
pip install langchain-openai
```

```
pip install langchain-huggingface
```

```
pip install -U langchain-chroma
```