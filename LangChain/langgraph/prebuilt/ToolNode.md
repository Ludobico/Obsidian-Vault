- [[#도구 정의|도구 정의]]
- [[#ToolNode 수동 호출해보기|ToolNode 수동 호출해보기]]
- [[#병렬 수행|병렬 수행]]
- [[#ChatModel과 함께 사용|ChatModel과 함께 사용]]


[원본 블로그](https://sean-j.tistory.com/entry/LangGraph-ToolNode)

LLM에 Tool을 binding 해서 LLM이 `tool_calls`를 생성했을 때, 적절한 arguments를 사용해 해당 tool을 실행하도록 하는 `ToolNode`에 대해 자세히 알아보자. 방금 작성한 그대로 로직을 작성할 수도 있지만([참고 - [LangGrpah] Tool Binding](https://sean-j.tistory.com/entry/LangGrpah-Tool-Binding)), LangGraph는 ToolNode를 사전 정의(pre-built)해서 제공한다.

내부적으로는 LLM에게 tool의 목록을 전달하고 (bind_tools), LLM이 사용자의 질문을 기반으로 tool 실행이 필요하다고 판단하면 해당 tool의 이름과 arguments를 반환한다. 그러면 해당 tool과 arguments로 함수를 실행하게 된다. 이때 `tool` list를 갖고, LLM이 반환한 `tool_calls`를 기반으로 함수를 실행할 수 있도록 노드로 구현한 것이 `ToolNode`다.

![[Pasted image 20250529154432.png]]

#### 도구 정의

먼저 파이썬 코드를 실행하는 `execute_python`과 location의 따라 서로 다른 문자열을 반환하는`get_weather`를 정의하자. 그리고 이 두 함수를 tool 콜백 함수를 사용해서 llm에 binding 하기 좋은 형태로 변환하자.

  
다음으로, 두 개의 tool을 list 형태로 만든 뒤, ToolNode를 초기화하자.

```python
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.prebuilt import ToolNode

@tool
def execute_python(code: str):
    """Call to excute python code."""
    return PythonAstREPLTool().invoke(code)

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["seoul", "busan"]:
        return "The temperature is 5 degrees and it's cloudy."
    else:
        return "The temperature is 30 degrees and it's sunny."

# Tool Node 초기화
tools = [execute_python, get_weather]
tool_node = ToolNode(tools=tools)
```

### ToolNode 수동 호출해보기

> ToolNode는 State의 messages list의 마지막 메세지가 tool_calls 인자가 있는지 없는지 여부로 tool을 호출할지를 결정한다.

일반적으로는 AIMessage를 수동으로 생성하지 않고, LangChain의 LLM이 생성하지만, 먼저 ToolNode를 수동으로 호출해 보자.

AIMessage에서 content는 보통 빈 문자열이 들어가고, `tool_calls` 속성에 `List[Dict]` 형태로, 호출할 도구의 이름, 인자, ID, 유형의 키 키값으로 들어간다.

```python
message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "seoul"},
            "id": "tool_call_id",
            "type": "tool_call"
        }
    ]
)

print(tool_node.invoke({"messages": [message_with_single_tool_call]}))
```

ToolNode는 `execute_python`과 `get_weather` 두 개의 tool을 갖고 있지만, invoke 한 결과는 ToolMessage로 반환되며, content에는 get_weather(location="seoul") 을 실행한 결과가 들어간다.

```python
{'messages': [ToolMessage(content="The temperature is 5 degrees and it's cloudy.", name='get_weather', tool_call_id='tool_call_id')]}
```

### 병렬 수행

AIMessage의 tool_calls 인자에 리스트로 여러 tool을 전달하면, ToolNode가 병렬적으로 도구 호출을 수행한다. 아래는 `execute_python`과 `get_weather` 두 가지 tool_call을 tool_calls 리스트에 전달했다. 그리고 결과는 예상대로 2개의 ToolMessage가 반환된다.

```python
message_with_multiple_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "busan"},
            "id": "tool_call_id_1",
            "type": "tool_call"
        },
        {
            "name": "execute_python",
            "args": {"code": "3 + 3"},
            "id": "tool_call_id_2",
            "type": "tool_call"
        }
    ]
)

print(tool_node.invoke({"messages": [message_with_multiple_tool_call]}))
```

```python
{'messages': [ToolMessage(content="The temperature is 5 degrees and it's cloudy.", name='get_weather', tool_call_id='tool_call_id_1'), ToolMessage(content='6', name='execute_python', tool_call_id='tool_call_id_2')]}
```

### ChatModel과 함께 사용

앞서 말한 것처럼, 일반적으로는 AIMessage를 수동으로 생성하지 않고, LangChain의 LLM이 생성한다. `bind_tools` 메소드를 호출해서 LLM에 도구를 인식시켜 줄 수 있다.

실제로는 아래와 같은 프롬프트가 들어간다! ([참고](https://python.langchain.com/v0.1/docs/use_cases/tool_use/prompting/))

```
"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""
```

그럼 이번에는 tool을 들고 있는 `llm_with_tools`를 정의하고, 서울의 현재 날씨를 물어보자. 그러면 LLM이 `get_weather` tool을 호출해야 한다고 판단하고, AIMessage에 `tool_calls` 인자가 채워져 반환된다.

```python
from langchain_openai import ChatOpenAI

llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools=tools)
llm_with_tools.invoke("seoul의 현재 날씨는 어때요?").tool_calls
```

```python
[{'name': 'get_weather', 
  'args': {'location': 'seoul'}, 
  'id': 'call_7TfQhajj4O3fM1FROdH4gAwC', 
  'type': 'tool_call'}]
```

따라서, 해당 반환값을 직접 ToolNode에 전달해 실행할 수 있다.

```python
tool_node.invoke({
    "messages": [llm_with_tools.invoke("seoul의 현재 날씨는 어때요?")]
})
```

```python
{'messages': [ToolMessage(content="The temperature is 5 degrees and it's cloudy.", name='get_weather', tool_call_id='call_jDy3hBVkBKHzHDkzC1UuzyOX')]}
```

