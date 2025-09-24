
- [[#Parameters|Parameters]]
- [[#Example code|Example code]]


<font color="#ffff00">AIMessage</font>는 **AI 모델이 생성한 응답 메시지**를 나타냅니다.

[[LangChain/LangChain|LangChain]] 에서는 모델의 raw output 과 함께, 표준화된 필드(예 : [[Tool]] call, usage metadata)를 추가합니다.

## Parameters

> content -> str or list\[str\]

- 메시지의 실제 텍스트 내용입니다.

```python
from langchain_core.messages import AIMessage

msg = AIMessage(content="서울은 한국의 수도입니다.")
print(msg.content)
# 출력: 서울은 한국의 수도입니다.
```

> additional_kwargs -> optional, dict

- 메시지에 연관된 추가 payload

```python
msg = AIMessage(
    content="독수리 부리는 노란색입니다.",
    additional_kwargs={"note": "생태 관련 정보"}
)
print(msg.additional_kwargs)
# 출력: {'note': '생태 관련 정보'}
```


> example -> optional, bool

- 예시 대화 여부를 표시합니다. 대부분은 무시됩니다.

```python
msg = AIMessage(content="예시 메시지", example=True)
print(msg.example)
# 출력: True
```

> id -> optional, str

- 메시지 고유 id 값입니다.

```python
msg = AIMessage(content="메시지", id="msg-001")
print(msg.id)
# 출력: msg-001
```

> name -> optional, str

- 메시지 고유의 이름입니다.

```python
msg = AIMessage(content="메시지", name="Assistant Reply")
print(msg.name)
# 출력: Assistant Reply
```

> response_metadata -> optional, dict

- 응답과 관련된 메타데이터입니다.

```python
msg = AIMessage(
    content="메시지",
    response_metadata={"model_name": "gpt-4o-mini", "finish_reason": "stop"}
)
print(msg.response_metadata)
# 출력: {'model_name': 'gpt-4o-mini', 'finish_reason': 'stop'}

```

> tool_call -> optional, list\[ToolCall\]

- 메시지와 연관된 tool call 입니다.

```python
msg = AIMessage(
    content="툴 호출 예시",
    tool_calls=[{"name": "search", "arguments": {"query": "독수리 부리 색"}}]
)
print(msg.tool_calls)
# 출력: [{'name': 'search', 'arguments': {'query': '독수리 부리 색'}}]

```

> type -> Literal['ai'], default : 'ai'

- 메시지의 타입입니다.

```python
msg = AIMessage(content="메시지")
print(msg.type)
# 출력: ai
```

> usage_metadata -> UsageMetadata

- 메시지의 토큰 사용량 등을 측정합니다.

```python
from langchain_core.messages import UsageMetadata

usage = UsageMetadata(input_tokens=12, output_tokens=8, total_tokens=20)
msg = AIMessage(content="독수리 부리는 노란 이유", usage_metadata=usage)
print(msg.usage_metadata.total_tokens)
# 출력: 20
```

## Example code

```python
llm = gpt()

question = "독수리 부리는 왜 노랄까?"

response = llm.invoke(question)
output_tokens = llm.get_num_tokens(response.content)
print(response)
print(output_tokens)
```

```
content='독수리의 부리가 노란색인 이유는 여러 가지가 있습니다. 첫째, 노란색은 독수리의 건강과 성숙도를 나타내는 신호로 작용할 수 있습니다. 밝고 선명한 색깔은 다른 독수리나 짝에게 자신의 건강 상태가 좋다는 것을 알리는 역할을 합니다.\n\n둘째, 노란색은 태양광에 잘 반사되어 독수리가 사냥을 할 때 시각적으로 유리할 수 있습니다. 또한, 노란색 부리는 독수리가 먹이를 잡거나 다룰 때 더 잘 보이게 하여 사냥에 도움이 될 수 있습니다.\n\n마지막으로, 색소의 종류와 분포에 따라 부리의 색깔이 결정되며, 이는 유전적 요인과 환경적 요인에 의해 영향을 받을 수 있습니다. 독수리의 부리 색깔은 종에 따라 다를 수 있으며, 각 종의 생태적 특성과 관련이 있습니다.' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_51db84afab', 'service_tier': 'default'} id='run--d9a4224f-9cfc-4c0a-9156-d4cee62d50c7-0'
209
```

