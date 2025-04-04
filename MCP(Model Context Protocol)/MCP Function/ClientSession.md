
`ClientiSession` 은 [[MCP(Model Context Protocol)]] 에서 **서버와의 통신 세션을 관리하는 핵심 클래스**입니다. **비동기 통신을 지원**하며, 서버와의 메시지 교환을 추상화합니다.

```python
class ClientSession(
):
    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception],
        write_stream: MemoryObjectSendStream[types.JSONRPCMessage],
        read_timeout_seconds: timedelta | None = None,
        sampling_callback: SamplingFnT | None = None,
        list_roots_callback: ListRootsFnT | None = None,
    ) -> None:
```


## Methods

### initilize()

```python
async def initialize(self)
```

서버와의 초기 연결을 설정하고 핸드셰이크를 수행합니다.
세션 시작 직후에 반드시 호출해야 합니다.

### call()

```python
async def call(
    self,
    method: str,      # 호출할 서버 메서드 이름
    params: Any = None # 메서드에 전달할 파라미터
) -> Any             # 서버로부터의 응답
```

서버에 특정 메서드를 호출하고 결과를 받아옵니다.

## Example code

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

async def basic_session_example():
    # 서버 파라미터 설정
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )
    
    # 세션 생성 및 사용
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 세션 초기화
            await session.initialize()
            
            # 서버 메서드 호출
            result = await session.call(
                method="echo",
                params={"message": "Hello Server!"}
            )
            print(f"서버 응답: {result}")
```

## Example multiple methods

```python
async def multiple_calls_example():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 수학 연산 메서드 호출
            calc_result = await session.call(
                method="calculate",
                params={"expression": "2 + 2"}
            )
            print(f"계산 결과: {calc_result}")
            
            # 상태 확인 메서드 호출
            status = await session.call(
                method="get_status"
            )
            print(f"서버 상태: {status}")
```

## Example error handling

```python
async def error_handling_example():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                
                # 존재하지 않는 메서드 호출 시도
                result = await session.call(
                    method="non_existent_method",
                    params={"data": "test"}
                )
            except ConnectionError as e:
                print(f"연결 오류: {e}")
            except Exception as e:
                print(f"메서드 호출 오류: {e}")
```

