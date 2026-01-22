- [[#Overview|Overview]]
- [[#Parameters|Parameters]]
- [[#Returns|Returns]]
- [[#Event 값|Event 값]]
- [[#이벤트의 계층 구조 이해|이벤트의 계층 구조 이해]]
- [[#실무 활용 팁|실무 활용 팁]]
- [[#Example : Tool calling이 포함된 그래프 실행|Example : Tool calling이 포함된 그래프 실행]]
	- [[#Example : Tool calling이 포함된 그래프 실행#`on_chain_start`|`on_chain_start`]]
	- [[#Example : Tool calling이 포함된 그래프 실행#`on_chain_start`|`on_chain_start`]]
	- [[#Example : Tool calling이 포함된 그래프 실행#`on_chat_model_start`|`on_chat_model_start`]]
	- [[#Example : Tool calling이 포함된 그래프 실행#`on_chat_model_stream`|`on_chat_model_stream`]]
	- [[#Example : Tool calling이 포함된 그래프 실행#`on_chain_end`|`on_chain_end`]]
- [[#example 2 : 값을 스트리밍으로 출력|example 2 : 값을 스트리밍으로 출력]]



## Overview

`astream_events()` 는 단순한 스트리밍 API가 아니라 [[LangGraph]] 에서 **실행 엔진이 발생시키는 모든 내부신호를 외부에서 관찰하기 위한 인터페이스** 입니다.

`graph.invoke()` 가 최종 결과만 반환하는 고수준 API 라면, `graph.astream_events()` 는 그래프가 실행되는 동안 **무슨 일이 일어나고 있는지**를 단계별로 전달하는 메서드입니다.

## Parameters

```python
async for event in graph.astream_events(
    input_data: dict,
    *,
    version: str = "v2",
    stream_mode: str = "values",
    config: Optional[dict] = None
):
```

> input_data -> dict

- 그래프에 전달할 입력 데이터입니다.

> version -> str

- 스트리밍 이벤트의 버전입니다. `v1` 또는 `v2` 를 지원하며 `v2` 는 구조화된 JSON 형태로 좀 더 깔끔하게 전달됩니다.

> stream_mode -> str

- `values` , `events`, `none` 중 하나로, 보통 `events` 모드에서 세부 이벤트를 볼 수 있습니다.

> config -> dict

- 실행 환경 옵션을 정의합니다.(특정 노드 비활성화 등)

## Returns

astream_events() 는 **비동기 제너레이터** 이므로,

```python
async for event in graph.astream_events(...):
```

형태로 반복문에서 사용해야 합니다.

각 `event` 는 딕셔너리 형태로 들어오며, 보통 이런 구조를 가집니다.

```json
{
    "event": "on_chat_model_stream",
    "data": {
        "chunk": {
            "content": "Hello, ",
            "metadata": {...}
        }
    },
    "node_name": "chat_model",
    "timestamp": "2025-10-31T06:14:23Z"
}
```

## Event 값

| 이벤트 그룹        | event 값                | 설명                    | name 예시                    | data 예시 (주요 키)                            |
| ------------- | ---------------------- | --------------------- | -------------------------- | ----------------------------------------- |
| **Chain**     | `on_chain_start`       | 그래프, 노드, 혹은 파이프라인 시작  | `LangGraph`, `agent`       | `{"input": {...}}`                        |
|               | `on_chain_stream`      | 노드나 파서가 중간 결과를 출력할 때  | `agent`, `StrOutputParser` | `{"chunk": "문자열"}` 혹은 `{"chunk": {...}}`  |
|               | `on_chain_end`         | 그래프나 노드의 실행 완료        | `agent`, `LangGraph`       | `{"output": {"answer": "..."}}`           |
| **Model**     | `on_chat_model_start`  | LLM(채트 모델) 호출 시작      | `ChatOpenAI`, `ChatGemini` | `{"input": {"messages": [...]}}`          |
|               | `on_chat_model_stream` | **LLM의 실시간 토큰 출력**    | `ChatOpenAI`               | `{"chunk": AIMessageChunk(content="..")}` |
|               | `on_chat_model_end`    | LLM 응답 생성 완료          | `ChatOpenAI`               | `{"output": AIMessage(content="...")}`    |
| **Tool**      | `on_tool_start`        | `@tool`로 정의된 함수 호출 시작 | `get_weather`              | `{"input": {"city": "Seoul"}}`            |
|               | `on_tool_end`          | 도구 함수 실행 완료           | `get_weather`              | `{"output": "서울은 현재 25도..."}`             |
| **Retriever** | `on_retriever_start`   | 벡터스토어 등에서 검색 시작       | `Retriever`                | `{"query": "서울 날씨"}`                      |
|               | `on_retriever_end`     | 검색 완료 및 문서 반환         | `Retriever`                | `{"documents": [Document(...)]}`          |
| **Parser**    | `on_parser_start`      | OutputParser가 해석을 시작함 | `PydanticOutputParser`     | `{"input": "..."}` (LLM의 원문 텍스트)          |
|               | `on_parser_end`        | 해석 완료 후 구조화된 데이터 반환   | `PydanticOutputParser`     | `{"output": IntentAnalysis(..)}`          |
| **Error**     | `on_error`             | 실행 중 예외 발생 시          | (에러 발생 컴포넌트 이름)            | `{"exception": "ValueError: ..."}`        |

## 이벤트의 계층 구조 이해

`astream_events`는 마치 양파 껍질처럼 **상위 체인이 하위 컴포넌트를 감싸는 구조**로 발생합니다. 이를 시각화하면 다음과 같습니다.

1. **`on_chain_start` (LangGraph)**: 전체 그래프의 시작
    
2. **`on_chain_start` (agent_node)**: 특정 노드의 시작
    
3. **`on_chat_model_start` (LLM)**: 노드 내부에서 모델 호출
    
4. **`on_chat_model_stream` (LLM)**: 실시간 토큰 생성 (반복)
    
5. **`on_chat_model_end` (LLM)**: 모델 답변 완료
    
6. **`on_chain_end` (agent_node)**: 노드 작업 종료
    
7. **`on_chain_end` (LangGraph)**: 전체 프로세스 종료
    

---

## 실무 활용 팁

- **`on_chain_stream` vs `on_chat_model_stream`**:
    - 파서가 적용된 깔끔한 텍스트만 원하면 `on_chain_stream` (단, 파서가 스트리밍을 지원해야 함)을 보세요.
        
    - 가장 빠른 반응성과 모델의 메타데이터(토큰 사용량 등)까지 원하면 `on_chat_model_stream`을 보세요.
        
- **`name` 필드의 활용**:
    - 동일한 노드 안에서 LLM을 두 번 호출한다면, `name`을 통해 어떤 LLM의 신호인지 구분할 수 있습니다. (예: `fast_llm` vs `main_llm`)
        
- **`tags` 필드 활용**:
    - 사용자님이 직접 체인에 `.with_config(tags=["my-tag"])`를 붙이면, 이벤트 발생 시 `tags` 컬럼에 해당 값이 포함되어 필터링이 훨씬 쉬워집니다.


## Example : Tool calling이 포함된 그래프 실행

이 예제는 사용자의 질문을 받고, 필요 시 도구를 호출하며, LLM이 답변을 생성하는 과정을 추적합니다.

```python
import asyncio
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from LLM.LLMs import gpt

# 간단한 도구 정의
@tool
def get_weather(city : str):
    """특정 지역의 날씨를 알려줍니다."""
    return f"{city}는 현재 25도입니다."

class State(TypedDict):
    messages : list

llm = gpt(temperature=0.7)
llm.bind_tools([get_weather])

async def call_model_node(state : State) -> State:
    response = llm.ainvoke(state['messages'])
    return {"messages" : response}

workflow = StateGraph(State)
workflow.add_node("agent", call_model_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
graph = workflow.compile()

async def main():
    inputs = {"messages" : [
        {
            "role" : "user",
            "content" : "서울 날씨 알려줘"
        }
    ]}

    async for event in graph.astream_events(inputs, version='v2'):
        kind = event['event']
        data = event['data']
        metadata = event['metadata']
        name = event["name"]
        tags = event['tags']

        print(f"\n[이벤트]: {kind}")
        print(f"\n[메타데이터]: {metadata}")
        print(f"\n[이름]: {name}")
        print(f"\n[태그]: {tags}")
        print("-"*80)

if __name__ == "__main__":
    asyncio.run(main())
```

```
[이벤트]: on_chain_start

[메타데이터]: {}

[이름]: LangGraph

[태그]: []
--------------------------------------------------------------------------------

[이벤트]: on_chain_start

[메타데이터]: {'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ('branch:to:agent',), 'langgraph_path': ('__pregel_pull', 'agent'), 'langgraph_checkpoint_ns': 'agent:45547dfb-1a42-85a4-4b18-d2d149a76f79'}

[이름]: agent

[태그]: ['graph:step:1']
--------------------------------------------------------------------------------

[이벤트]: on_chain_stream

[메타데이터]: {'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ('branch:to:agent',), 'langgraph_path': ('__pregel_pull', 'agent'), 'langgraph_checkpoint_ns': 'agent:45547dfb-1a42-85a4-4b18-d2d149a76f79'}

[이름]: agent

[태그]: ['graph:step:1']
--------------------------------------------------------------------------------

[이벤트]: on_chain_end

[메타데이터]: {'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ('branch:to:agent',), 'langgraph_path': ('__pregel_pull', 'agent'), 'langgraph_checkpoint_ns': 'agent:45547dfb-1a42-85a4-4b18-d2d149a76f79'}

[이름]: agent

[태그]: ['graph:step:1']
--------------------------------------------------------------------------------

[이벤트]: on_chain_stream

[메타데이터]: {}

[이름]: LangGraph

[태그]: []
--------------------------------------------------------------------------------

[이벤트]: on_chain_end

[메타데이터]: {}

[이름]: LangGraph

[태그]: []
--------------------------------------------------------------------------------
```


### `on_chain_start`

전체 그래프 실행이 시작될 때 발생합니다.

### `on_chain_start`

특정 노드가 실행될 때 발생합니다.


### `on_chat_model_start`

노드 내부에서 LLM 호출이 시작될 때 발생합니다.
    

### `on_chat_model_stream`

**가장 중요한 이벤트입니다.** 모델이 답변을 생성할 때마다 발생합니다.


### `on_chain_end`

그래프의 모든 처리가 끝나고 최종 상태(State)를 반환할 때 발생합니다.


## example 2 : 값을 스트리밍으로 출력

```python
import asyncio
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from LLM.LLMs import gpt


class State(TypedDict):
    messages : list

llm = gpt(temperature=0.7)

async def call_model_node(state : State) -> State:
    response = await llm.ainvoke(state['messages'])
    return {"messages" : response}

workflow = StateGraph(State)
workflow.add_node("agent", call_model_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
graph = workflow.compile()

async def main():
    inputs = {"messages" : [
        {
            "role" : "user",
            "content" : "서울 날씨 알려줘"
        }
    ]}

    async for event in graph.astream_events(inputs, version='v2'):
        kind = event['event']
        data = event['data']

        if kind == 'on_chat_model_stream':
            print(data['chunk'].content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

이 예제는 LangGraph에서 **LLM의 토큰 스트리밍을 테스트하는 가장 단순한 형태**의 그래프 구성입니다. 그래프는 단 하나의 노드만을 가지며, 해당 노드는 LLM을 호출한 뒤 바로 종료됩니다.

