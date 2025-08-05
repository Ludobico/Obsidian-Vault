[[LangGraph]] 에서는 [[Python]] Dict 기반의 상태(state)를 노드 간에 전달하면서 점진적으로 채워나가는 구조가 가능하고, `TypedDict` 는 그 상태의 형을 힌트로 제공합니다.

`TypedDict` 는 <font color="#ffff00">total=False</font> 가 아니면 **모든 키가 필수로 간주**됩니다.

```python
class State(TypedDict):
    start: bool
    task: int
    model: str
    question: str
    answer: str
    thinking_mode: bool
```

이건 정적 분석 기준으로 모든 키가 필수입니다.

즉, `graph.invoke(stat)` 시점에 이 필드들을 모두 넣지 않으면 IDE나 type checker는 경고를 줄 수 있습니다.

## TypedDict(total=False)

```python
class State(TypedDict, total=False):
    start: bool
    task: int
    model: str
    question: str
    answer: str
    thinking_mode: bool
```

이렇게 하면 IDE나 타입 검사 도구도 **모든 필드는 Optional**로 간주합니다.

