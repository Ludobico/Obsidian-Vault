[[Asyncio]] 의 `loop.run_in_executor()` 메서드는 일반적인 콜백 기반의 <font color="#ffff00">동기 코드를 비동기 코드로 사용할 수 있도록 지원</font> 합니다. 이 메서드를 사용하면 동기코드를 블로킹하지 않고도 비동기 이벤트 루프에서 실행할 수 있습니다.

일반적인 사용방법은 다음과 같습니다.

```python
import asyncio
from typing import Awaitable

def sync_function(arg: str) -> str:
  return print("hello", arg)

async def async_function(arg : str) -> Awaitable:
  loop = asyncio.get_event_loop()
  result = await loop.run_in_executor(None, sync_function, arg)

if __name__ == "__main__":
  asyncio.run(async_function("world"))
```

```
hello world
```

여기서 `loop.run_in_executor()` 는 두 가지 주요 파라미터를 가지고 있습니다.

> executor -> ThreadPoolExecutor or ProcessPoolExecutor
- 동기 코드를 실행할 때 사용할 executor를 지정합니다. `None` 을 전달하면 디폴트로 `ThreadPoolExecutor` 가 사용됩니다.

> func -> Callable
- 실행할 동기 함수 또는 메서드를 지정합니다.

`run_in_executor()` 를 사용하면 CPU-bound 작업과 같은 블로킹 작업을 비동기 코드 내에서 실행할 수 있으며, 이를 통해 이벤트 루프가 블로킹되지 않고 계속 실행됩니다. 이는 I/O 바운드 작업과 CPU 바운드 작업을 효과적으로 다루기 위한 방법 중 하나입니다.

