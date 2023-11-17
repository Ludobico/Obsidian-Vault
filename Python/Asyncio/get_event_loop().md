`asyncio.get_event_loop()` 함수는 현재 실행 중인 스레드에서의 asyncio 이벤트 루프를 반환하는 함수입니다. [[Asyncio]] 는 비동기 코드를 실행하기 위한 라이브러리로, 이벤트 루프는 비동기 작업을 관리하고 실행하는 역할을 합니다.

여러 스레드에서 asyncio 를 사용하는 경우, 각 스레드에서 별도의 이벤트 루프가 필요합니다.

`asyncio.get_event_loop()` 함수는 <font color="#ffff00">현재 스레드에 대한 이벤트 루프를 반환</font>하며, 만약 현재 스레드에 이벤트 루프가 없으면 새로운 이벤트 루프를 생성합니다.

간단한 사용 예시를 살펴보겠습니다.

```python
import asyncio

async def my_coroutine():
  print("Coroutine is running")

if __name__ == "__main__":
  loop = asyncio.get_event_loop()
  loop.run_until_complete(my_coroutine())
  loop.close()
```

```
Coroutine is running
```

1. `get_event_loop()` 를 사용하여 현재 스레드의 이벤트 루프를 얻습니다.
2. `run_until_complete()` 를 사용하여 비동기 함수를 실행합니다.
3. `close()` 함수로 이벤트 루프를 종료합니다.

`get_event_loop()` 를 호출할 때, <font color="#ffff00">이벤트 루프가 현재 스레드에 존재하지 않는다면 새로운 이벤트 루프를 생성하고, 이미 존재한다면 그것을 반환</font>합니다.

