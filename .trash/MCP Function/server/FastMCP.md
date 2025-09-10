<font color="#ffff00">FastMCP</font> 는 [[MCP(Model Context Protocol)]] 를 구현한 서버 인터페이스이며, [[Python]] 비동기 프로그래밍을 활용해 연결 관리, 프로토콜 준수, 메시지 라우팅 등을 처리하는 핵심 컴포넌트입니다.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")
```

- <font color="#ffff00">FastMCP</font> 는 이름(`"dmeo"`) 를 인자로 받아 **서버 인스턴스를 생성**합니다. 이 이름은 서버를 식별하나 로깅, 디버깅에 사용됩니다.
