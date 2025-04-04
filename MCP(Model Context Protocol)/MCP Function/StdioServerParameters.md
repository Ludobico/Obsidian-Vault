- [[#Example code|Example code]]
- [[#Example code using all parameters|Example code using all parameters]]
- [[#Example intergration with stdio_client|Example intergration with stdio_client]]
- [[#Python|Python]]
- [[#Node.js|Node.js]]


`StdioServerParameters` 는 표준 입출력(stdio)을 통해 통신하는 [[MCP(Model Context Protocol)]] 서버의 **실행 파라미터를 정의하는 클래스**입니다. 서버 프로세스를 시작하고 관리하는 데 필요한 설정을 담고 있습니다.

```python
class StdioServerParameters:
    def __init__(
        self,
        command: str,                         # 실행할 명령어
        args: List[str],                      # 명령어 인자 리스트
        cwd: Optional[str] = None,            # 작업 디렉토리 (선택)
        env: Optional[Dict[str, str]] = None  # 환경 변수 (선택)
    )
```

> command -> str

실행할 서버 프로그램의 명령어입니다.

"python", "node", "java" 등으로 입력합니다.

> args -> List\[str\]

명령어에 전달할 인자들의 리스트입니다. 서버 스크립트 경로나 설정 파일 경로등을 입력합니다.

> cwd -> Optional\[str\]

서버 프로세스가 실행될 작업 디렉토리입니다. 기본값은 `None` 으로 현재 디렉토리입니다.

> env -> Optional\[Dict\[str, str\]\]

서버 프로세스에 전달할 환경 변수입니다.

---

`StdioServerParameters` 를 사용할때 `commands` 와 `args` 는 반드시 제공해야 하며, 경로를 지정할 때는 절대 경로를 사용하는 것이 안전합니다. 또한 환경 변수는 현재 시스템의 환경 변수를 덮어쓰므로 주의해서 사용해야 하고, 작업 디렉토리(cwd)를 지정할 때는 해당 디렉토리가 존재하는지 확인해야 합니다. 이러한 방식으로 `StdioServerParameters` 를 사용하면 MCP 서버의 실행 환경을 세밀하게 제어할 수 있습니다.

## Example code

```python
from mcp import StdioServerParameters

# 가장 기본적인 사용
basic_params = StdioServerParameters(
    command="python",
    args=["server.py"]
)
```

## Example code using all parameters

```python
# 모든 파라미터를 사용하는 상세 설정
detailed_params = StdioServerParameters(
    command="python",
    args=[
        "math_server.py",
        "--port=8000",
        "--debug=true"
    ],
    cwd="/path/to/server/directory",
    env={
        "PYTHONPATH": "/custom/python/path",
        "DEBUG": "1",
        "SERVER_MODE": "development"
    }
)
```

## Example intergration with stdio_client

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

async def server_communication_example():
    # 서버 파라미터 설정
    server_params = StdioServerParameters(
        command="python",
        args=["math_server.py"],
        env={"PYTHONPATH": "./lib"}
    )
    
    # stdio_client와 함께 사용
    async with stdio_client(server_params) as (read, write):
        # 서버와의 통신 로직
        await write({"type": "request", "data": "hello"})
        async for message in read:
            print(f"Received: {message}")
```

## Python

```python
# Python 서버 실행
python_server_params = StdioServerParameters(
    command="python",
    args=[
        "math_server.py",
        "--config=config.json"
    ],
    env={"PYTHONPATH": "./lib"}
)
```

## Node.js

```python
node_server_params = StdioServerParameters(
    command="node",
    args=[
        "server.js",
        "--env=production"
    ],
    env={
        "NODE_ENV": "production",
        "PORT": "3000"
    }
)
```

