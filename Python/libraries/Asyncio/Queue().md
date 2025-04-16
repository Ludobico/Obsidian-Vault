[[Asyncio]] 라이브러리에서 제공하는 `Queue` 모듈은 비동기 큐를 구현하는 데 사용됩니다. 이 모듈은 `asyncio.Queue()` 클래스를 제공하여 여러 `asyncio` <font color="#ffff00">태스크 간의 데이터를 안전하게 교환하는데 사용</font>됩니다. 비동기 큐는 `put` 메서드를 사용하여 데이터를 큐에 추가하고, `get` 메서드를 사용하여 큐에서 데이터를 추출할 수 있습니다. 큐에 데이터가 없을 때 `get` 을 호출하면 태스크가 블록되어 데이터가 들어올 때까지 대기합니다.

간단한 사용 예제를 살펴보겠습니다.

```python
import asyncio
from typing import Awaitable

async def producer(queue, item):
  print(f"Producing {item}")
  await asyncio.sleep(1)
  await queue.put(item)

async def consumer(queue):
  while True:
    item = await queue.get()
    print(f"Consuming {item}")
    await asyncio.sleep(1)

async def main():
  q = asyncio.Queue()

  producer_task = asyncio.create_task(producer(q, "item_1"))
  consumer_task = asyncio.create_task(consumer(q))

  await asyncio.gather(producer_task,asyncio.sleep(0.5) ,consumer_task)


if __name__ == "__main__":
  asyncio.run(main())
```

```
Producing item_1
Consuming item_1
```

이 예제에서는 `asyncio.Queue()` 를 사용하여 `producer` 함수가 아이템을 생성하여 큐에 넣고, `consumer` 함수가 큐에서 아이템을 소비합니다. `asyncio.gather` 를 사용하여 이들을 동시에 실행하고, `asyncio.sleep()` 을사용하여 비동기 작업을 시뮬레이션합니다.

`asyncio.Queue` 의 주요 메서드는 다음과 같습니다.

> asyncio.Queue.put(item)
- 큐에 아이템을 추가합니다.

> asyncio.Queue.get()
- 큐에서 아이템을 가져옵니다. 큐가 비어 있으면 아잍메이 들어올 때까지 대기합니다.

> asyncio.Queue.qsize()
- 큐의 현재 크기를 반환합니다.

> asyncio.Queue.empty()
- 큐가 비어있는지 여부를 반환합니다.

> asyncio.Queue.full()
- 큐가 가득 찼는지 여부를 반환합니다.

> asyncio.Queue.maxsize
- 큐의 최대 크기를 나타내는 속성입니다.

