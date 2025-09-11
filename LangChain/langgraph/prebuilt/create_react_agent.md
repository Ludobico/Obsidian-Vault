- [[#Main Features|Main Features]]
- [[#Parameters|Parameters]]
- [[#Langchain의 create_react_agent|Langchain의 create_react_agent]]
- [[#Langgraph의 create_react_agent|Langgraph의 create_react_agent]]
- [[#둘이 뭔 차이에요?|둘이 뭔 차이에요?]]
- [[#정리|정리]]


```python
create_react_agent(model: Union[str, LanguageModelLike], tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode], *, prompt: Optional[Prompt] = None, response_format: Optional[Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]] = None, pre_model_hook: Optional[RunnableLike] = None, state_schema: Optional[StateSchemaType] = None, config_schema: Optional[Type[Any]] = None, checkpointer: Optional[Checkpointer] = None, store: Optional[BaseStore] = None, interrupt_before: Optional[list[str]] = None, interrupt_after: Optional[list[str]] = None, debug: bool = False, version: Literal['v1', 'v2'] = 'v1', name: Optional[str] = None) -> CompiledGraph
```

<font color="#ffff00">create_react_agent</font> 는 LLM과 [[Tool]] 을 결합하여, 질문을 처리하기 위해 **추론(reasoning)** 과 **액션(acting)** 을 반복적으로 수행하는 에이전트를 생성합니다. ReAct 패러다임은 LLM이 문제를 해결하기 위해 단계별로 생각하고(추론), 필요한 경우 외부 도구를 호출하여(액션) 정보를 얻은 뒤 이를 바탕으로 최종 답변을 도출하는 방식입니다. 이 함수는 복잡한 그래프 구조를 직접 정의하지 않아도, 간단한 설정으로 ReAct 에이전트를 만들 수 있게 해줍니다.

## Main Features

- [[LangGraph]] 에서 복잡한 노드와 엣지를 직접 설계하지 않고도 ReAct 에이전트 생성
- LLM이 도구를 호출하도록 자동으로 관리
- 메시지 히스토리와 같은 상태를 유지하며 반복적인 추론-행동 루프를 실행

## Parameters

> model -> Union\[str, LanguageModel\]

- ReAct 에이전트가 사용할 LLM으로, tool calling을 지원하는 모델이어야 합니다.

> tools -> Union\[Sequence\[Union\[BaseTool, Callable\]\], ToolNode\]

- 에이전트가 사용할 Tool list 또는 ToolNode 인스턴스입니다. 빈 리스트를 제공하면 Tool 없이 LLM만 실행합니다.

```python
from typing import Dict
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.llms import OpenAI
# 1. @tool 데코레이터를 사용한 방법
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """특정 위치의 날씨를 반환합니다."""
    # 실제로는 날씨 API를 호출하는 로직이 들어갈 것입니다
    return f"{location}의 날씨는 맑음입니다."
```

```python
# 2. Tool 클래스를 직접 사용한 방법
def search_function(query: str) -> str:
    """검색을 수행합니다."""
    return f"{query}에 대한 검색 결과입니다."

search_tool = Tool(
    name="Search",
    func=search_function,
    description="웹에서 정보를 검색할 때 사용하는 도구"
)
```

```python
# 3. ToolNode를 사용한 방법
from langchain.tools import ToolNode

# 도구들을 리스트로 모음
tools = [get_weather, search_tool]

# ToolNode로 감싸기
tool_node = ToolNode(tools)
```

> prompt -> Optional, prompt, default : None

- LLM에 전달할 초기 프롬프트입니다. 다양한 형태로 제공이 가능합니다.
	- str : [[SystemMessage]] 로 변환되어 메시지 목록 맨 앞에 추가
	- [[SystemMessage]] : 메시지 목록 맨 앞에 추가
	- Callable : 상태를 받아 LLM 입력을 생성
	- [[runnables]] : 상태를 받아 LLM 입력을 생성하는 객체

> response_format -> optional, StructuredResponseSchema, tuple

- 최종 출력 형식을 정의하는 스키마입니다.

```
- an OpenAI function/tool schema,
- a JSON Schema,
- a TypedDict class,
- or a Pydantic class.
- a tuple (prompt, schema), where schema is one of the above.
    The prompt will be used together with the model that is being used to generate the structured response.
```

<font color="#ffc000">Important</font>
모델이 `.with_structured_output` 을 지원해야합니다.

> pre_model_hook -> optional, default : None

- LLM 호출 전에 실행되는 녿, 메시지 정리(요약, 삭제) 등에 사용됩니다.

```json
# At least one of `messages` or `llm_input_messages` MUST be provided
{
    # If provided, will UPDATE the `messages` in the state
    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), ...],
    # If provided, will be used as the input to the LLM,
    # and will NOT UPDATE `messages` in the state
    "llm_input_messages": [...],
    # Any other state keys that need to be propagated
    ...
}
```

<font color="#ffc000">Important</font>
1. `messages` 또는 `llm_input_messages` 중 하나를 사용해야합니다.

> checkpointer -> optional, default : None

- 상태를 저장하는 체크포인터입니다. **대화 메모리 유지에 사용**됩니다.
- 주로 `MemorySaver()` 를 사용합니다.
- `thread_id` 와 함께 사용하여 대화의 지속성을 제공합니다.

> store -> optional, default : None

- 여러 스레드 간의 데이터를 저장합니다.
- 다중 사용자/대화 간 데이터를 공유합니다.

> debug -> optional, default : False

- 디버그 모드 활성화 여부입니다. 실행 로그를 출력하는데 사용됩니다.

> version -> optional, Literal\["v1", "v2"\], default : "v1"

- 그래프의 버전입니다.
	- `v1` : 단일 메시지의 모든 도구 호출을 병렬 처리합니다.
	- `v2` : 도구 호출별로 `Send API`를 사용해 분산 처리합니다.

> name -> optional, str, default : None

- 그래프의 이름입니다. **다중 에이전트 시스템에서 서브그래프로 사용 시** 유용합니다.
- 예 : "weather_agent"


## Langchain의 create_react_agent

[공식 API 문서](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html#create-react-agent)

Langchain에서는 `create_react_agent` 함수를 통해서 Agent를 만들 수 있고

prompt를 통해서 Agent가 어떤 Chain Of Thought 를 진행할지 기술하게 됩니다.

```python
#홈페이지 Example
from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent

from langchain_core.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)
model = OpenAI()
tools = ...

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": "hi"})

# Use with chat history
from langchain_core.messages import AIMessage, HumanMessage
agent_executor.invoke(
    {
        "input": "what's my name?",
        # Notice that chat_history is a string
        # since this prompt is aimed at LLMs, not chat models
        "chat_history": "Human: My name is Bob\nAI: Hello Bob!",
    }
)
```

그래서 사용자의 prompt작성 수준에 따라서 Agent의 Tools Call능력이나, Reasoning능력이 달라지게 됩니다.

하지만 API문서에서 보면, 아래와 같은 문구를 확인할 수가 있습니다.

```txt
Warning

This implementation is based on the foundational ReAct paper but is older and not 
well-suited for production applications. For a more robust and feature-rich 
implementation, we recommend using the create_react_agent function from the LangGraph 
library. See the [reference doc](https://langchain-
ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.crea
te_react_agent) for more information.
```

Langchain Agent는 Production 용도에는 적합하지 않다는 내용이네요.

Langchain의 경우 prompt를 이용해서 Agent가 Tool을 사용하게 유도합니다.

이 덕분에 Tool Call이 학습되지 않은 모델에서도 Tool을 사용할 수 있다는 장점이 있습니다.

## Langgraph의 create_react_agent

[공식 API 문서](https://langchain-ai.github.io/langgraph/reference/agents/)

Langgraph의 경우 동일한 이름의 `create_react_agent` 함수를 제공합니다.

대신 차이점을 보이는게

Langgraph의 공식 문서에서는

prompt입력에 대한 가이드 라던가, 복잡한 prompt를 입력하라고 안내해주지 않습니다.

```python
#공식예제
#https://langchain-ai.github.io/langgraph/agents/agents/#2-create-an-agent
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## 둘이 뭔 차이에요?

| 비고            | Langchain                       | Langgraph                     |
| ------------- | ------------------------------- | ----------------------------- |
| prompt        | Reasoing과정 직접 작성                | 단순한 Agent의 역할부여               |
| ReAct의구현      | prompt로 정의+Langchain내부 구현       | Langgraph로 구현                 |
| Function call | prompt를 이용해서 LLM이 Tool을 사용하게 유도 | LLM 학습 당시 구현한 Tool Call API사용 |
| Tool 사용 가능 모델 | 모든 모델                           | Tool Call이 가능한 Model          |
두 라이브러리의 실제 vLLM 또는 OpenAI로 보내지는 prompt를 확인하기 위해서 아래 코드를 사용합니다.

```python
#langchain
import logging
import requests
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#logging을 통해서 debbuging진행
class LoggingChatOpenAI(ChatOpenAI):
    def _call(self, prompt: str, stop=None) -> str:
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Stop: {stop}")
        response = super()._call(prompt, stop)
        logger.debug(f"Response: {response}")
        return response


# Use the LoggingChatOpenAI class instead of ChatOpenAI
model = LoggingChatOpenAI(
    model="MyModel",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="mytoken", 
    base_url="http://myvllmservice/v1",
)

# Define a very simple tool function that returns the current time
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
    
# List of tools available to the agent
tools = [
    Tool(
        name="Weather",  # Name of the tool
        func=get_weather,  # Function that the tool will execute
        # Description of the tool
        description=get_weather.__doc__,
    ),
]

# Initialize a ChatOpenAI model
from langchain_core.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "tell me what time is it now"})

# Print the response from the agent
print("response:", response)
```

```python
#langgraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LoggingChatOpenAI(ChatOpenAI):
    def _call(self, prompt: str, stop=None) -> str:
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Stop: {stop}")
        response = super()._call(prompt, stop)
        logger.debug(f"Response: {response}")
        return response

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Use the LoggingChatOpenAI class instead of ChatOpenAI
model = LoggingChatOpenAI(
    model="MyModel",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="mytoken", 
    base_url="http://myvllmservice/v1",
)
# MCP Server 설정(이경우 Client.py파일과 Server.py파일이 같은 위치에 있음)
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["weather_server_stdio.py"],
)
from langchain_core.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

# MCP Python SDK를 활용해서 MCP Server를 사용함
async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # MCP Server에 있는 Tool리스트를 가지고 옵니다.
            tools = await load_mcp_tools(session)

            # Tools와 Model을 사용해서 Agent 생성
            agent = create_react_agent(
                model = model,
                tools = tools,
                prompt = template
            )
            
            # Agent한테 Query를 진행해, Agent가 알아서 tools을 사용하게 만듭니다.
            agent_response = await agent.ainvoke({"messages": "what's time is it now?. i'm live in seoul city. find about city seoul"})
            return agent_response

# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
```

```python
#weather_server_stdio.py
from mcp.server.fastmcp import FastMCP
import datetime

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

이제 코드를 실행시키고, 디버깅 로그를 확인해봅니다.

```
#langchain을 사용한 경우
DEBUG:openai._base_client:Request options: {'method': 'post', 'url': 
'/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-
75c082d1-3005-4aa5-9924-8b88f7afb4b3', 'json_data': {'messages': [{'content': 'Answer 
the following questions as best you can. You have access to the following 
tools:\n\nTime(city: str) -> str - Get weather for a given city.\n\nUse the following 
format:\n\nQuestion: the input question you must answer\nThought: you should always 
think about what to do\nAction: the action to take, should be one of [Time]\nAction 
Input: the input to the action\nObservation: the result of the action\n... (this 
Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the 
final answer\nFinal Answer: the final answer to the original input 
question\n\nBegin!\n\nQuestion: how about weather in seoul?\nThought:', 'role': 
'user'}], 'model': 'Gemma-3-27b-it', 'stop': ['\nObservation'], 'stream': True, 
'temperature': 0.0}}
```

prompt만 따로 빼서보면

```
Answer the following questions as best you can. You have access to the following tools:
Time(city: str) -> str - Get weather for a given city.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Time]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer\nFinal Answer: the final answer to the original input question

Begin!

Question: how about weather in seoul?
Thought:'
```

requests를 보내기 전에 이미 prompt부분이 Agent정의때 사용된 `{input}, {tools}, {tools_names}, {agent_scratchppad}` 가 이미 포맷팅이 된 것을 확인할 수 있습니다.

실제로 requests를 보내기 이전에 prompt 포맷팅이 진행되는 것음

langchain의 create_react_agent 소스코드에서도 확인할 수가 있습니다.

```python
#https://python.langchain.com/api_reference/_modules/langchain/agents/react/agent.html#create_react_agent
def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: Optional[AgentOutputParser] = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: Union[bool, list[str]] = True,
) -> Runnable:
    
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    
```

반면 Langgraph를 확인해보면

```
DEBUG:openai._base_client:Request options: {'method': 'post', 'url': 
'/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-
2ce447ef-a0e2-434c-94a2-5577d9ffcbb7', 'json_data': {'messages': [{'content': 'Answer 
the following questions as best you can. You have access to the following 
tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you 
must answer\nThought: you should always think about what to do\nAction: the action to 
take, should be one of [{tool_names}]\nAction Input: the input to the 
action\nObservation: the result of the action\n... (this Thought/Action/Action 
Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal 
Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: 
{input}\nThought:{agent_scratchpad}', 'role': 'system'}, {'content': "what's time is 
it now?. i'm live in seoul city. find about city seoul", 'role': 'user'}], 'model': 
'Gemma-3-27b-it', 'stream': False, 'temperature': 0.0, 'tools': [{'type': 'function', 
'function': {'name': 'get_current_time', 'description': 'useful tools when you need 
to know current time', 'parameters': {'properties': {'city': {'type': 'string'}}, 
'required': ['city'], 'type': 'object'}}}]}}
```

Langgraph는 prompt가 포맷팅 되지 않고 그대로 전달되게 됩니다.

여기서 Langchain과의 큰 차이점은 requests의 parameter로 tools가 전달되게 됩니다.

```json
'tools': [
    {'type': 'function',
    'function': {
        'name': 'get_current_time',
        'description': 'useful tools when you need to know current time', 
        'parameters': {
            'properties': {
                'city': {'type': 'string'}
            }, 'required': ['city'],
                'type': 'object'
            }
        }
    }
]
```

## 정리

- Langchain : Client Side에서 prompt를 처리함
    
- Langgraph : LLM Service Side에서 prompt를 처리함
    

외부 환경에서 상용 서비스(ChatGPT, Gemini, Claude)를 이용할 경우

langgraph로 tool call API를 이용할 수가 있습니다.

그렇기 때문에 이런경우는 정상적으로 Tool을 사용하는 Agent를 구현할 수 있습니다.

> 특히나 요즘 MCP가 대세인데, MCP를 사용하기 위해서는 Langgraph를 이용하는 것이 좋습니다,

**하지만 vLLM등을 이용하는 Private LLM환경에서는**

Tool Call이 허용되지 않는 모델이나, vLLM 설정에 따라 Tool Call이 불가능 할 수 있습니다.

이런 경우에 Agent를 구현하기 위해서 Langchain을 사용한 Agent구현을 고려해 볼 수 있습니다.

