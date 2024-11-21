`SystemMessagePromptTemplate` 은 [[LangChain/LangChain|LangChain]] 에서 제공하는 템플릿 중 하나로, **시스템 메시지를 정의하여 템플릿으로 제작**하는데 사용합니다.

```python
from langchain.prompts.chat import SystemMessagePromptTemplate, ChatPromptTemplate

# 시스템 메시지 템플릿 생성
system_template = "You are a {persona} expert who explains complex topics in a {style} way."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


# ChatPromptTemplate 조합
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
])

# 프롬프트 포맷팅
prompt = chat_prompt.format_messages(
    persona="scientific",
    style="simple"
)

chat_prompt.pretty_print()

print(f"\n{prompt}")
```

```
================================ System Message ================================

You are a {persona} expert who explains complex topics in a {style} way.

[SystemMessage(content='You are a scientific expert who explains complex topics in a simple way.', additional_kwargs={}, response_metadata={})]
```

