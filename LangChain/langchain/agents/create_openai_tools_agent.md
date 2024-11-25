`create_openai_tools_agent` 는 OpenAI의 최신 function calling 매커니즘을 사용하는 에이전트 생성 클래스입니다. 일반적으로 [[create_tool_calling_agent]] 보다 더 나은 성능을 보입니다.

**OpenAI의 최신 GPT 모델을 LLM으로 사용할 것을 권장** 합니다.

## Parameters

> llm -> BaseLanguageModel

- 에이전트가 사용할 언어 모델입니다.
- **최신 GPT 모델을 권장**합니다.

> tools -> Sequence[BaseTool]

- 에이전트가 접근 가능한 Tool list 입니다.

> prompt -> [[ChatPromptTemplate]]

- 에이전트의 행동을 정의하는 프롬프트 템플릿입니다.

> strict -> optional, bool, Default : False

- 에이전트가 도구를 사용할 때의 제약 조건을 설정합니다.
	- `True` : 에이전트가 반드시 tool을 호출하며 사용이 강제됩니다.
	- `False` : 에이전트가 tool을 선택적으로 호출하며, 필요하지 않으면 tool을 호출하지 않습니다.


## Example code

```python
from langchain.agents import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import SearchTool, CalculatorTool

# 도구 생성
search_tool = SearchTool()
calc_tool = CalculatorTool()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4-turbo")

# 프롬프트 템플릿 생성 (예시)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")
])

# OpenAI Tools Agent 생성
agent = create_openai_tools_agent(
    llm=llm, 
    tools=[search_tool, calc_tool], 
    prompt=prompt
)
```

