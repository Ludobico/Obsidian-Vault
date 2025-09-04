
[[LangGraph]] 는 [[State]]를 공유하는 그래프 구조에서, 각 노드가 수행할 작업은 **업데이트할 키만 반환**하는 형태로 정의됩니다. 노드에서 반환된 값은 LangGraph 내부에서 기존 상태와 병합됩니다.

- 만약 노드에서 아무런 값을 반환하지 않으면 (`return None` 또는 생략), LangGraph는 **업데이트 없음**으로 처리하고 **이전 상태를 그대로 다음 노드로 전달**합니다.

- 이 덕분에 노드가 파일 쓰기 같은 부수 효과만 수행할 경우에도 상태가 유지되어 워크플로우가 끊기지 않습니다.

- 반대로 병렬 노드(<font color="#ffff00">fanin fanout</font> 구조) 에서 동시에 같은 키를 업데이트하려 하면 LangGraph는 `INVALID_CONCURRENT_GRAPH_UPDATE`  에러를 발생시키며, 충돌을 방지하도록 설계되어 있습니다.

## Example

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

# 1. 상태 정의
class State(TypedDict):
    text: str
    processed: Optional[str]
    saved: Optional[bool]

# 2. 노드 정의
def loader_node(state: State):
    # 초기 입력값을 로드
    return {"text": "Hello LangGraph"}

def processor_node(state: State):
    # 일부 값만 업데이트
    return {"processed": state["text"].upper()}

def saver_node(state: State):
    # return 없음 → 업데이트 없이 부수 효과만 수행
    with open("output.txt", "w") as f:
        f.write(state["processed"])
    # return None 은 state를 그대로 전달

def final_node(state: State):
    # 여기서 state는 여전히 존재
    print("최종 상태:", state)

# 3. 그래프 빌드
builder = StateGraph(State)
builder.add_node("loader", loader_node)
builder.add_node("processor", processor_node)
builder.add_node("saver", saver_node)
builder.add_node("final", final_node)

builder.add_edge(START, "loader")
builder.add_edge("loader", "processor")
builder.add_edge("processor", "saver")
builder.add_edge("saver", "final")
builder.add_edge("final", END)

graph = builder.compile()

# 4. 실행
graph.invoke({})

```

