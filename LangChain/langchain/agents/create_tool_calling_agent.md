`create_tool_calling_agent` 는 [[LangChain/LangChain|LangChain]] 에서 사용되는 함수로, **도구를 호출하는 Agent를 생성** 하는 함수입니다. 기본적으로 모델과 Tool List를 입력으로 받아 에이전트를 초기화합니다.

```python
from langchain.agents import create_tool_calling_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate

# 1. 도구 정의
@tool
def search_tool(query: str) -> str:
    """인터넷 검색을 수행하는 도구"""
    return f"Search result for: {query}"

@tool
def calculator_tool(expression: str) -> str:
    """수학 계산을 수행하는 도구"""
    return str(eval(expression))

# 2. 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools."),
    ("human", "{input}"),
])

# 3. LLM 초기화
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 4. 에이전트 생성
agent = create_tool_calling_agent(
    llm=llm,
    tools=[search_tool, calculator_tool],
    prompt=prompt
)

# 5. 에이전트 실행기 생성
agent_executor = AgentExecutor(
    agent=agent, 
    tools=[search_tool, calculator_tool], 
    verbose=True
)

# 6. 에이전트 실행
result = agent_executor.invoke({"input": "Calculate 25 * 4"})
```

## Parameters

> llm -> BaseLanguageModel

- 에이전트가 사용할 언어 모델을 지정합니다.

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")  # GPT-4 모델을 사용
```

> tools -> Sequence, BaseTool

- 에이전트가 사용할 수 있는 Tool List 입니다. `BaseTool` 클래스의 객체들로 이루어진 시퀀스를 전달합니다.

```python
from langchain_core.tools import Tool

# 검색 도구
def search_tool(query: str) -> str:
    return f"Searching for {query}"

# 계산기 도구
def calculate(expression: str) -> str:
    return str(eval(expression))

tools = [
    Tool(name="search_tool", func=search_tool, description="Search online."),
    Tool(name="calculator", func=calculate, description="Perform calculations.")
]
```

> prompt -> [[ChatPromptTemplate]]

- 에이전트가 사용할 프롬프트 템플릿입니다. [[ChatPromptTemplate]] 객체를 사용하여, 에이전트가 모델에 전달할 텍스트 메시지를 어떻게 형성할지를 정의합니다.
	- 템플릿은 [[MessagesPlaceholder]] 를 포함하고 있으며, 실제 실행 시 변수 값으로 대체됩니다.

> message_formatter -> Callable

- 메시지 포매터 함수입니다. 이 함수는 `AgentAction` 객체와 그 결과로 얻은 Tools의 출력을 받아서, 이를 [[BaseMessage]] 형식의 메시지로 변환합니다.

