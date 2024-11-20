[[LangChain/LangChain|LangChain]] 의 `BaseMessage` 는 **메시지의 기본 구조를 정의** 하는 기초 클래스입니다.

```
langchain_core.messages.base.BaseMessage
```

> content -> str

- 메시지의 실제 내용을 저장

> type -> str

- 메시지의 유형을 정의 (예 : human, ai, system 등)

> additional_kwargs -> dict, optional

- 추가적인 메타데이터 정의

```python
class HumanMessage(BaseMessage):
    type: str = "human"

class AIMessage(BaseMessage):
    type: str = "ai"

class SystemMessage(BaseMessage):
    type: str = "system"
```

