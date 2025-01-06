
`RunnableWithMessageHistory` 는 [[LangChain/LangChain|LangChain]] 에서 메시지 기록(History)를 관리하는 [[runnables]] 래퍼 클래스입니다.

## Features

- **채팅 메시지 히스토리 관리**

- 세션 기반 대화 컨텍스트 유지

- 메시지 기록 읽기/쓰기 처리

## Parameters

> get_session_history 

- 새로운 `BaseChatMessageHistory` 을 반환하는 함수로, `session_id` 를 인자로 받아 해당하는 chat history 인스턴스를 반환합니다.

> input_messages_key

- Input 값이 dict 형태일 경우 지정합니다.

```python
result = chain.invoke({"quesion": user_input})
```

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
```

> output_messages_key

- Output 값이 dict 형태일 경우 지정합니다.

> history_messages_key

- 히스토리 메시지용 별도 키가 필요할 때 사용합니다.
- [[MessagesPlaceholder]] 에서 지정한 값을 주로 사용합니다.
- 현재 deprecated된 [[ConversationBufferMemory]] 의 `memory_key` 파라미터와 동일한 기능입니다.


## Example

```python
from operator import itemgetter
from typing import List

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | ChatAnthropic(model="claude-2")

chain_with_history = RunnableWithMessageHistory(
    chain,
    # Uses the get_by_session_id function defined in the example
    # above.
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foo"}}
))
```

