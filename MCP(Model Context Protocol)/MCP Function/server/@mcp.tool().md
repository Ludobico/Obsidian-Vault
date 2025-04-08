<font color="#ffff00">@tool()</font> 데코레이터는 [[FastMCP]] 서버에서 제공하는 기능으로, LLM과 같은 외부 시스템이 서버를 통해 특정 작업을 수행할 수 있도록 설계된 매커니즘입니다. 이를 통해 계산을 수행하거나 외부 API 호출과 같은 동작을 정의할 수 있습니다.

[[LangChain/LangChain|LangChain]] 의 [[@tool]] 과 유사한 점이 많습니다.
## Similiarities with langchain tool

1. LLM Intergration
	- 둘 다 LLM이 호출할 수 있는 함수를 정의합니다. 예를 들어, 사용자가 "BMI 계산해줘"라고 하면 LLM이 해당 도구를 실행합니다.
	- 함수의 **docstring을 활용해 도구의 용도를 설명**하며, LLM이 이를 이해하고 적절히 호출할 수 있도록 돕습니다.

2. 작업 수행
	- 단순 데이터 제공이 아닌 계산, 외부 API 호출 등 부수 효과를 일으키는 작어블 수행하도록 설계되었습니다.

3. 데코레이터 기반
	- `@mcp.tool()` 과 `@tool()` 모두 [[Decorator]] 를 활용해 **함수를 도구로 등록**합니다.


## Differences with langchain tool

1. 