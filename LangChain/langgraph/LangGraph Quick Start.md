
## Setup

첫 번째로, 패키지들을 설치하고 환경을 설치합니다. 여기의 예시에서는 GPT API를 사용합니다.

```bash
pip install langgraph langsmith langchain langchain-openai
```

```python
from config.getenv import GetEnv
from langchain_openai import ChatOpenAI

env = GetEnv()
api_key = env.get_openai_api_key

def chat_gpt(temperature : float = 0.1, model : str = 'gpt-4o-mini', streaming : bool = True):
    """
    You import a thousand GPT models every day. How 'bout you make this one do 'em all?
    """
    llm = ChatOpenAI(temperature=temperature, api_key=api_key, model=model, streaming=streaming)
    return llm
```

## Build a basic chatbot

먼저 [[langgraph]] 를 사용하여 간단한 챗봇을 만들어 보겠습니다. 이 챗봇은 사용자 메시지에 직접적으로 응답하는 방식으로 동작합니다. 단순한 예제지만, langgraph 로 구축하는 핵심 개념을 설명하기에 충분할 것입니다. 이 섹션이 끝나면 기본적인 챗봇을 완성하게 될 것입니다.

우선 **StateGraph** 를 생성합니다. StateGraph 객체는 챗봇의 구조를 <font color="#00b050">state machine</font> 으로 정의합니다. 여기에서 **Node**를 추가해 [[llms]] 과 챗봇이 호출할 수 있는 함수들을 나타내고, **Edge** 를 추가하여 챗봇이 이러한 함수들 사이를 어떻게 전환해야 하는지 지정할 것입니다.

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```

이 Graph는 두 가지 주요 작업을 처리할 수 있습니다.

1. 각 노드는 현재 상태(Current state)를 입력으로 받아 상태를 업데이트하는 출력을 생성합니다.
2. 메시지 업데이트는 기존 메시지 리스트를 덮어쓰는 대신, [[typing]] 의 Annotated 문법과 함께 사용하는 `add_messages` 함수 덕분에 기존 리스트에 추가됩니다.

다음으로 `chatbot` 노드를 추가합니다. 이 노드는 현재 작업의 일부를 나타내며, 일반적인 파이썬 함수로 구현됩니다.

```python
llm = chat_gpt()

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
```

`chatbot` 노드 함수가 현재 State를 입력으로 받고, 업데이트된 메시지 리스트를 **"messages" 키를 포함하는 딕셔너리를 반환하는 방식**에 주목하세요, 이것이 Langgraph 노드 함수의 기본 패턴입니다.

`add_messages` 함수는 State에 있는 기존 메시지에 LLM의 응답 메시지를 추가합니다.

다음으로, START 지점을 추가하세요. 이는 <font color="#ffff00">그래프가 실행될 때 어디서 시작해야 하는지</font>를 알려줍니다.

```python
graph_builder.add_edge(START, 'chatbot')
```

마찬가지로, 종료 지점을 설정하세요. 이는 그래프에게<font color="#ffff00"> 이 노드가 실행될 때 언제든 종료할 수 있다</font>고 지시합니다.

```python
graph_builder.add_edge("chatbot", END)
```

마지막으로 그래프를 실행할 수 있어야 합니다. 이를 위해 `compile()` 메서드를 호출하세요. 이는 State 에서 사용할 수 있는 <font color="#00b050">CompiledGraph</font>를 생성합니다.

```python
graph = graph_builder.compile()
```

그래프를 **시각화**하려면 `get_graph` 메서드와 `draw_ascii` 또는 `draw_png` 같은 draw 메서드 중 하나를 사용하세요. 이 draw 메서드들은 각각 추가적인 dependency가 필요합니다.

```python
from PIL import Image
import io

graph_png_data = graph.get_graph().draw_mermaid_png()
graph_img = Image.open(io.BytesIO(graph_png_data))
graph_img.show()
```

![[tmpbr_9cevh.png]]

