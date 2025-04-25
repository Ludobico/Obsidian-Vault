
# Understanding add_messages in Langgraph

[[LangGraph]]에서 대화 기록을 관리할 때 `add_messages` 라는 함수가 핵심 역할을 합니다. 이 함수가 뭔지, **왜 특정 포맷을 따라야**하는지, 그리고 문자열 반환 문제(예 : "A", "B", "C" 등)을 어떻게 처리하는지 이해할 수 있게 쉽게 정리했습니다.

## What is a add_messages?

`add_messages` 는 LangGraph에서 **대화 기록(`messages`)을 업데이트하는 도구**입니다. 사용자가 질문하고 AI가 답변할 때, 이 대화를 순서대로 쌓아서 관리해 줍니다. 비유하자면, 대화 노트에 새 메시지를 추가하는 펜 같은 개념입니다.

- 이 필드는 <font color="#ffff00">공식적으로</font>는 `Annotated[List, add_messages]` 로 정의됩니다.

```python
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List, add_messages]
```

## Why Does it require a Specific Format?

Langgraph 에서는 대화가 **특정 포맷**으로 저장됩니다. 이 포맷은

```
{"role": "user", "content": "질문"}
```

```
{"role": "assistant", "content": "답변"}
```

형태입니다. `add_messages` 가 이 포맷을 기대하기 때문에, **메시지를 만들 때 이 규칙을 따라야 합니다.**

