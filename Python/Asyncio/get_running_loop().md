`get_running_loop()` 함수는 [[Asyncio]] 에서 <font color="#ffff00">현재 실행 중인 이벤트 루프를 반환하는 함수</font>입니다. 이 함수는 [[Python]] 3.7 이상에서 사용할 수 있습니다.

`get_running_loop()` 을 사용하면 현재 실행 중인 코드가 어떤 이벤트 루프에서 실행 중인지를 확인할 수 있습니다. 이 함수를 호출하려면 반드시 <font color="#ffff00">비동기 함수 내에서 호출</font>되어야 합니다. 즉, `async def` 로 선언된 함수 내에서만 사용 가능합니다.

간단한 사용 예제를 살펴보겠습니다.

```python
import asyncio
from typing import Awaitable

async def my_coroutine() -> Awaitable[str]:
  loop = asyncio.get_running_loop()
  print("Running in loop :", loop)

async def main() -> Awaitable:
  await my_coroutine()

if __name__ == "__main__":
  asyncio.run(main())
```

```
Running in loop : <ProactorEventLoop running=True closed=False debug=False>
```

위 코드에서 `my_coroutine()` 함수 내에서 `asyncio.get_running_loop` 를 호출하여 현재 실행 중인 이벤트 루프를 얻고 있습니다. 이 함수를 사용하면 현재 코드가 어떤 이벤트 루프에서 실행중인지를 명시적으로 확인할 수 있습니다.

`get_running_loop()` 를 사용할 때 주의할 점은 반드시 비동기 함수 내에서 사용해야 하며, 이 함수를 호출하려면 현재 이벤트 루프가 존재해야 합니다. 따라서 `asyncio.run()` 같이 이벤트 루프를 생성하고 비동기 함수를 실행하는 메인 함수가 필요합니다.

