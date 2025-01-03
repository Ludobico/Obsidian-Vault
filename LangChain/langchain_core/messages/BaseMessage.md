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

> id -> optional

- 메시지의 고유 식별자입니다. ex ) `msg-12345`

> name -> optional
- 메시지의 사람이 읽을 수 있는 이름입니다.
- 이 필드는 사용 여부가 모델 구현에 따라 다릅니다.
- ex ) `UserQuery`

> response_metadata -> optional
- 응답과 관련된 메타데이터를 담는 딕셔너리입니다.

```python
class HumanMessage(BaseMessage):
    type: str = "human"

class AIMessage(BaseMessage):
    type: str = "ai"

class SystemMessage(BaseMessage):
    type: str = "system"
```

