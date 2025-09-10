FastMCP 서버 클래스는 [[MCP(Model Context Protocol)]] 애플리케이션을 구축하기 위한 핵심 컴포넌트로 <font color="#f79646">tools, resources, prompts</font> 를 관리합니다. 이 클래스는 애플리케이션의 주요 컨테이너 역할을 하며, MCP 클라이언트와의 통신을 담당합니다.

## Create Server

서버를 생성하는 과정은 간단합니다. 일반적으로 서버 이름을 지정하여 클라이언트 애플리케이션이나 로그에서 쉽게 식별할 수 있도록 합니다.

```python
from fastmcp import FastMCP

# Create a basic server instance
mcp = FastMCP(name="MyAssistantServer")

# You can also add instructions for how to interact with the server
mcp_with_instructions = FastMCP(
    name="HelpfulAssistant",
    instructions="""
        This server provides data analysis tools.
        Call get_average() to analyze numerical data.
    """,
)
```

### Parameters

> name -> str, Default : "FastMCP"

- 서버의 이름

> instructions -> str

- 서버와 상호작용하는 방법을 설명하는 문자열로 클라이언트가 서버의 목적과 사용 가능한 기능을 이하는 데 도움을 줍니다.

> auth -> OAuthProvider | TokenVerifier

- HTTP 기반 전송을 보호하기 위한 auto provider 입니다.

> lifespan -> AsyncContextManager

- 서버 시작 및 종료 로직을 정의하는 비동기 컨텍스트 매니저

> tools -> list[Tool | Callable]

- 서버에 추가할 도구 목록, `@mcp.tool` 데코레이터 대신 프로그래밍 방식으로 도구를 제공할 때 사용합니다.

> include_tags -> set[str]

- 지정된 태그와 일치하는 컴포넌트만 노출

> exclude_tags -> set[str]

- 지정된 태그와 일치하는 컴포넌트만 숨김

> on_duplicate_tools -> Literal["error", "warn", "replace"] default:"error"

- 중복된 도구를 처리하는 방식

> on_duplicate_resources -> Literal["error", "warn", "replace", default:"warn"

- 중복된 리소스 등록을 처리하는 방식

> on_duplicate_prompts -> Literal["error", "warn", "replace"], default:"replace"

- 중복된 프롬프트 등록을 처리하는 방식

> include_fastmcp_meta -> bool, default:"True"

컴포넌트 응답에 [[FastMCP]] 메타데이터를 포함할지 여부


