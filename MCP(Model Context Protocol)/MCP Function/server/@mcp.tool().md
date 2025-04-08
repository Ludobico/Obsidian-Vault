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

1. 프레임워크와 생태계
	- `@mcp.tool()`
		- [[FastMCP]] 라는 특정 서버 프레임워크에 속하며, MCP 프로토콜을 기반으로 동작합니다.
		- 서버 환경에서 연결 관리, 메시지 라우팅, lifespan과 통합되어 있습니다.
		- 주로 서버-클라이언트 구조에서 LLM이 서버의 기능을 호출하는 데 초점이 맞춰져 있습니다.
	- `@tool()`
		- [[LangChain/LangChain|LangChain]] 프레임워크에서 제공하며, LLM 중심의 워크플로우와 통합됩니다.
		- 서버가 아닌 파이썬 환경에서 독립적으로 실행되며, 주로 LLM의 로컬 또는 클라우드 기반 workflow에 사용됩니다.
		- Langchain의 agent나 [[create_tool_calling_agent]] 매커니즘과 연계됩니다.
		
2. 컨텍스트와 실행 환경
	- `@mcp.tool()`
		- FastMCP 서버의 컨텍스트를 활용할 수 있습니다.
		- 비동기 서버 환경에 최저고하되어 있으며, 동기/비동기 함수 모두 지원합니다.
		- 실행은 서버에서 이루어지며, 클라이언트(LLM) 가 서버에 요청을 보내 호출합니다.

	- `@mcp.tool()`
		- Langchain의 도구는 주로 로컬 또는 LLM 실행환경에서 동작하며, **별도의 서버가 필요하지 않습니다.**
		- [[Tool]] 객체로 변환되어 LLM의 에이전트가 직접 호출하며, 컨텍스트는 Langchain의 [[AgentExecutor]] 나 Chain에서 관리됩니다.
		- 비동기 지원은 있지만, 주로 동기 함수에 초점이 맞춰져 있습니다.

