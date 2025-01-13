
## Overview

Langgraph는 [[LangChain/LangChain|LangChain]] Ecosystem 의 일부로, **State 기반**의 LLM 애플리케이션을 구축하기위한 프레임워크입니다. 이는 LCEL([[runnables]]) 을 기반으로 하며, 복잡한 상태 관리와 워크플로우를 쉽게 구현할 수 있도록 도와줍니다.

## StateGraph

- StateGraph 는 LangGraph의 핵심 컴포넌트입니다.
- **노드와 엣지로 구성된 그래프 구조**를 가집니다.
- 각 노드는 특정 상태나 작업을 나타냅니다.
- 엣지는 상태 간의 전환 조건을 정의합니다.

## State Management

- State는 애플리케이션의 현재 컨텍스트를 저장합니다.
- State는 불변(immutable)하며, 각 전환마다 새로운 State가 생성됩니다.
- State는 Dictionary 형태로 관리됩니다.