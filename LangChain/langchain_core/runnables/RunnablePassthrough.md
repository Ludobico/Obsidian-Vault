- [[#Memory implementation|Memory implementation]]
- [[#RAG Implementation|RAG Implementation]]


`RunnablePassthrough` 는 [[LangChain/LangChain|LangChain]] 에서 **데이터를 변경하지 않고 그대로 전달**하는 역할을 하는 클래스입니다.

`RunnablePassthrough` 는 입력된 데이터를 그대로 반환하기 때문에 데이터를 변환하거나 수정할 필요가 없는 경우, 파이프라인의 특정 단계를 건너뛰어야 하는 경우, 디버깅 또는 테스트 목적으로 데이터 흐름을 모니터링해야 하는 경우에 주로 사용됩니다.

```python
from Utils.highlight import highlight_print

from langchain_core.runnables import  RunnablePassthrough

runnable = RunnablePassthrough()
result = runnable.invoke("Hellow world")
highlight_print(result)

result = runnable.invoke({"num" : 1})
highlight_print(result)
```

```
--------------------------------------------------------------------------------
Hellow world
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
{'num': 1}
--------------------------------------------------------------------------------
```



## Memory implementation

```python
RunnablePassthrough.assign(
    chat_history=lambda x: memory.chat_memory.messages
)
```

- `RunnablePassthrough`  는 입력 데이터를 그대로 출력으로 전달하는 [[LangChain/langchain/langchain|langchain]] 의 기본 컴포넌트입니다.

- `assgin()` 메서드는 입력데이터를 수정하거나 새 키를 추가할 때 사용합니다.
- 여기서는 `lambda x: memory.chat_memory.messages` 를 통해 `chat_history` 의 값을 동적으로 설정합니다.


## RAG Implementation

```python
retrieval_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)
result = retrieval_chain.invoke("What does the dog want to eat?")
```

이 코드가 동작하는 이유는 다음과 같은 **LCEL의 내부 동작 방식** 때문입니다.

```python
{
    "context": retriever,
    "question": RunnablePassthrough(),
}
```

LCEL은 전체 체인의 입력 (`str` : `What does the dog want to eat?`) 을 이 딕셔너리 구조에 맞게 각 항목의 [[runnables]] 에 동일하게 전달합니다.

즉

```
"context": retriever -> retriever.invoke("What does the dog want to eat?") → List[Document]

"question": RunnablePassthrough() -> RunnablePassthrough().invoke("What does the dog want to eat?") → "What does the dog want to eat?"
```

이렇게 되어서 딕셔너리 형태로

```python
{
    "context": [Document(...), ...],
    "question": "What does the dog want to eat?"
}
```

이 구조가 다음 단계인 `prompt` 로 넘어가게 됩니다.

