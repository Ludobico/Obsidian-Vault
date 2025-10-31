- [[#Parameters|Parameters]]
- [[#Returns|Returns]]
- [[#Examples|Examples]]


<font color="#ffff00">astream_events()</font> 는 [[LangGraph]] 실행 중 발생하는 중간 이벤트(노드 실행 시작, 모델 응답 스트림, 실행 종료 등)을 실시간으로 받아볼 수 있게 해줍니다.

- `graph.invoke()` : 실행 결과만 반환
- `graph.astream_events()` : 실행 과정을 스트림으로 하나씩 반환

## Parameters

```python
async for event in graph.astream_events(
    input_data: dict,
    *,
    version: str = "v2",
    stream_mode: str = "values",
    config: Optional[dict] = None
):
```

> input_data -> dict

- 그래프에 전달할 입력 데이터입니다.

> version -> str

- 스트리밍 이벤트의 버전입니다. `v1` 또는 `v2` 를 지원하며 `v2` 는 구조화된 JSON 형태로 좀 더 깔끔하게 전달됩니다.

> stream_mode -> str

- `values` , `events`, `none` 중 하나로, 보통 `events` 모드에서 세부 이벤트를 볼 수 있습니다.

> config -> dict

- 실행 환경 옵션을 정의합니다.(특정 노드 비활성화 등)

## Returns

astream_events() 는 **비동기 제너레이터** 이므로,

```python
async for event in graph.astream_events(...):
```

형태로 반복문에서 사용해야 합니다.

각 `event` 는 딕셔너리 형태로 들어오며, 보통 이런 구조를 가집니다.

```json
{
    "event": "on_chat_model_stream",
    "data": {
        "chunk": {
            "content": "Hello, ",
            "metadata": {...}
        }
    },
    "node_name": "chat_model",
    "timestamp": "2025-10-31T06:14:23Z"
}
```

| event 값              | 설명                     |
| -------------------- | ---------------------- |
| on_chain_start       | 그래프(혹은 특정 노드)의 실행이 시작됨 |
| on_chain_end         | 그래프(혹은 특정 노드)의 실행이 끝남  |
| on_tool_start        | 도구(tool) 호출 시작         |
| on_tool_end          | 도구 호출 종료               |
| on_chat_model_stream | LLM의 토큰 스트리밍 출력        |
| on_chat_model_end    | 모델 출력이 끝남              |
| on_error             | 오류 발생시                 |

## Examples

```python
async for event in graph.astream_events(input_data, version="v2"):
    # 모델 출력 중간 스트림을 실시간으로 출력
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"].content
        if chunk:
            print(chunk, end="", flush=True)

    # 전체 그래프 완료 시점
    elif event["event"] == "on_chain_end":
        print("\n[Graph execution completed]")

```

