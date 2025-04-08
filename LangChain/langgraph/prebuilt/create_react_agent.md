- [[#Main Features|Main Features]]
- [[#Parameters|Parameters]]


```python
create_react_agent(model: Union[str, LanguageModelLike], tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode], *, prompt: Optional[Prompt] = None, response_format: Optional[Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]] = None, pre_model_hook: Optional[RunnableLike] = None, state_schema: Optional[StateSchemaType] = None, config_schema: Optional[Type[Any]] = None, checkpointer: Optional[Checkpointer] = None, store: Optional[BaseStore] = None, interrupt_before: Optional[list[str]] = None, interrupt_after: Optional[list[str]] = None, debug: bool = False, version: Literal['v1', 'v2'] = 'v1', name: Optional[str] = None) -> CompiledGraph
```

<font color="#ffff00">create_react_agent</font> 는 LLM과 [[Tool]] 을 결합하여, 질문을 처리하기 위해 **추론(reasoning)** 과 **액션(acting)** 을 반복적으로 수행하는 에이전트를 생성합니다. ReAct 패러다임은 LLM이 문제를 해결하기 위해 단계별로 생각하고(추론), 필요한 경우 외부 도구를 호출하여(액션) 정보를 얻은 뒤 이를 바탕으로 최종 답변을 도출하는 방식입니다. 이 함수는 복잡한 그래프 구조를 직접 정의하지 않아도, 간단한 설정으로 ReAct 에이전트를 만들 수 있게 해줍니다.

## Main Features

- [[langgraph]] 에서 복잡한 노드와 엣지를 직접 설계하지 않고도 ReAct 에이전트 생성
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


