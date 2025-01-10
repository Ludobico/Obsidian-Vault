- [[#Example code|Example code]]
- [[#Parameters|Parameters]]


`AgentExecutor` 는 LLM 기반 에이전트가 [[tools]] 를 사용하여 문제를 해결할 수 있도록 실행 환경을 제공합니다.

## Example code

```python
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import Tool

# Define LLM and tools
llm = ChatOpenAI(model="gpt-4")
tools = [Tool(name="calculator", func=lambda x: str(eval(x)), description="Calculates expressions.")]

# Define the agent
agent = ZeroShotAgent(llm=llm, tools=tools)

# Create the executor
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=True)

# Run the agent
response = executor({"input": "What is 2+2?"})
print(response)

```
## Parameters

> agent -> BaseSingleActionAgent, BaseMultiActionAgent, Runnable

- 에이전트는 입력을 받아서 계획을 수립하고 필요한 작업(calling tools)을 결정합니다.

```python
from langchain.agents import ZeroShotAgent

agent = ZeroShotAgent(llm=my_llm, tools=my_tools)
executor = AgentExecutor(agent=agent, tools=my_tools)

```

> tools -> BaseTool

- 에이전트가 호출할 Tool list 입니다.
- 각 tool은 특정 작업을 수행하는 함수로 검색, 계산, 데이터 호출 등의 기능을 제공합니다.

```python
from langchain_core.tools import Tool

tools = [
    Tool(name="calculator", func=lambda x: eval(x), description="Performs calculations.")
]
executor = AgentExecutor(agent=my_agent, tools=tools)
```

> callbacks -> Callbacks

- 에이전트 실행 동안 호출되는 콜백 핸들러를 정의합니다.
- 콜백은 실행 시작(`on_chain_start`), 종료(`on_chain_end`) , 또는 오류 발생(`on_chain_error`) 시 호출됩니다.

> early_stopping_method -> str, Default : "force"

- 에이전트가 작업을 종료하는 방식을 설정합니다.
	- "force" : 최대 시간/반복 횟수에 도달하면 작업 중지 메시지를 반환합니다.
	- "generate" : 반복 제한 도달 후, 최종 답변을 생성하여 반환합니다.

> handle_parsing_errors -> Union[bool, str, Callable], Default : False

- 에이전트 출력 파싱 중 발생하는 오류를 처리하는 방법입니다.
	- `False` : 예외 발생시 raise error로 처리합니다.
	- `True` : 오류 내용을 모델의 Observation 으로 전달합니다.
	- 문자열 : 해당 문자열을 모델의 Observation 으로 전달합니다.
	- 함수 : 예외를 처리하는 커스텀 로직을 정의합니다.

> max_execution_time -> optional, float, Default : None

- 최대 실행 시간을 설정합니다, 이 시간이 초과되면 루프가 중단됩니다.
- `None` 으로 설정하면 시간 제한 없이 실행됩니다.

> max_iterations -> optional, int, Default : 15 

- 에이전트가 수행할 최대 작업 반복 횟수를 설정합니다.
- 기본값은 15이며, `None` 으로 설정하면 무한 루프가 발생할 수 있습니다.

```python
executor = AgentExecutor(agent=my_agent, tools=tools, max_iterations=10)
```

> memory -> optional, BaseMemory

- 에이전트 실행 시 사용할 메모리 객체입니다.
- 메모리는 이전 대화나 작업 결과를 저장하고, 이후 작업에 사용할 수 있게합니다.
- 예를 들어, 대화형 에이전트의 경우 사용자와의 이전 대화 내용을 저장합니다.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
executor = AgentExecutor(agent=my_agent, tools=tools, memory=memory)

```

> metadata -> optional, Dict[str, Any], Default : None

- 에이전트와 관련된 메타데이터를 설정합니다.

> return_intermediate_steps -> bool, Default : False

- 에이전트가 작업을 완료한 후 중단 단계 경로를 반환할지 여부를 결정합니다.
- `True` 로 설정하면 각 단계의 작업 내용과 결과를 반환합니다.
- `False` 로 설정하면 최종 결과만 반환합니다.

> tage -> optional, List[str]

- 특정 실행을 식별하기 위해 사용되는 태그 리스트입니다.

> trim_intermediate_steps -> Union[int, Callable[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]], Default : -1

- 반환되는 중간 단계를 자르거나 요약하는 방법을 설정합니다.
- `-1` : 자르지 않습니다.
- 정수 : 반환할 단계의 수를 설정합니다.
- 함수 : 중간 단계를 직접 처리하는 커스텀 로직을 정의합니다.

> verbose -> optional, bool

- 디버깅 정보 출력 여부를 설정합니다.
- `True` 로 설정하면 실행 중간에 로그가 출력됩니다.

