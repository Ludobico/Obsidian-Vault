<font color="#ffff00">StateGraph</font> 클래스는 **노드들이 공유된 상태**를 읽고 쓰는 방식으로 상호작용하는 그래프입니다. 이 클래스에서 각 노드는 State를 입력으로 받고, Partial을 반환하는 형식을 가집니다.

## Parameters

> state_schema -> Any, Default : None

- 상태를 정의하는 스키마 클래스입니다. State 객체의 구조를 정의하고 검증하는 데 사용됩니다.

> config_schema -> optional, Any, Default : None

- config를 정의하는 스키마 클래스입니다. API에서 설정 가능한 매개변수를 설정하는데 사용됩니다.

## Example

```python
class State(TypedDict):
    messages : Annotated[list, add_messages]

graph_builder = StateGraph(State)
```

