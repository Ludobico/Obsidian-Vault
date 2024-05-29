[[LangChain/LangChain|LangChain]] 의 `ConversationBufferMemory` 는 **대화 메모리를 저장하는 버퍼를 제공하는 클래스**입니다. 이 클래스는 주어진 키워드 인수를 통해 입력 데이터를 파싱하고 검증하여 새로운 모델을 생성합니다. 입력 데이터가 유효한 모델을 형성할 수 없을 경우 `ValidationError` 를 발생시킵니다.

```python
class langchain.memory.buffer.ConversationBufferMemory
```

이 클래스의 파라미터는 다음과 같습니다.

> ai_prefix -> str, Default : 'AI'
- 인공지능 응답 앞에 붙는 접두어를 지정합니다.

> chat_memory -> Optional, BaseChatMessageHistory
- 대화 메시지 히스토리를 저장하는 객체를 지정합니다.

> human_prefix -> str, Default : 'Human'
- 사용자 입력 앞에 붙는 접두어를 지정합니다.

> input_key -> Optional[str] = None
- 입력 키를 지정합니다.

> output_key -> Optional[str] = None
- 출력 키를 지정합니다.

> return_messages -> bool, Default : False
- `True`로 설정할 경우, 저장된 메시지를 반환합니다.

