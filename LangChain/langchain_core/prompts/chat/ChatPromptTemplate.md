- [[#example code|example code]]
- [[#Implement|Implement]]


`ChatPromptTemplate` 은 [[LangChain/LangChain|LangChain]] 에서 chat model을 위한 프롬프트 템플릿을 정의하는데 사용되는 클래스입니다. 기존의 [[PromptTemplate]] 과는 다르게 **채팅 모델의 특성을 고려하여 설계**되었습니다.

주요 특징은 다음과 같습니다.

1. 채팅 모델은 일반적으로 유저와 AI간의 대화 형식으로 입력을 받습니다. `ChatPromptTemplate` 은 이러한 대화 형식을 지원하여 프롬프트를 구성할 수 있습니다.

2. 프롬프트는 [[AIMessage]] 와 [[HumanMessage]] 객체로 구성됩니다. 이를 통해 AI와 사용자의 메시지를 명확하게 구분할 수 있습니다.

3. 각 메시지 객체에는 `role` 속성이 있어, AI 또는 사용자의 역할을 지정할 수 있습니다. 이를 통해 프롬프트에 맥락 정보를 추가할 수 있습니다.

4. `ChatPromptTemplate` 은 여러 개의 메시지 객체를 조합하여 하나의 프롬프트를 만들 수 있습니다.

## example code

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?"
    }
)
```

```
# Output:
# ChatPromptValue(
#    messages=[
#        SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
#        HumanMessage(content='Hello, how are you doing?'),
#        AIMessage(content="I'm doing well, thanks!"),
#        HumanMessage(content='What is your name?')
#    ]
#)
```


## Implement
---

```python
from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
messages
```

```
[SystemMessage(content='이 시스템은 천문학 질문에 답변할 수 있습니다.'),
 HumanMessage(content='태양계에서 가장 큰 행성은 무엇인가요?')]
```

이렇게 생성된 메시지 리스트는 대화형 인터페이스나 언어 모델과의 상호작용을 위한 입력으로 사용될 수 있습니다. 각 메시지는 `role` (메시지를 말하는 주체, system 또는 user) 과 `content` (메시지 내용) 속성을 포함합니다. 이 구조는 시스템과 사용자 간의 대화 흐름을 명확하게 표현하며, 언어 모델이 이를 기반으로 적절한 응답을 생성할 수 있도록 돕습니다.

```python
chain = chat_prompt | llm | StrOutputParser()

chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
```

```
태양계에서 가장 큰 행성은 목성입니다. 목성은 태양 주위를 도는 행성 중에서 가장 크고 질량도 가장 많이 가지고 있습니다.
```

```python
  def prompt_chat_template():
    chat_prompt = ChatPromptTemplate.from_messages([
      SystemMessagePromptTemplate.from_template("이 시스템은 주어진 질문에 답변할 수 있습니다."),
      HumanMessagePromptTemplate.from_template("{user_input}, {test_input}")
    ])
    return chat_prompt
```

```python
    prompt = GemmaWithLangchain.prompt_chat_template()
	question = "태양계에서 가장 큰 행성은 무엇인가요?"
    chains = LLMChain(llm=langchain_pipeline, prompt=prompt, verbose=True, output_parser=StrOutputParser())
    print(chains.invoke({"user_input" : question, "test_input" : "이 문장은 테스트문장입니다."}))

```

```
# verbose
Prompt after formatting:
System: 이 시스템은 주어진 질문에 답변할 수 있습니다.
Human: 태양계에서 가장 큰 행성은 무엇인가요?, 이 문장은 테스트문장입니다.

# result
{'user_input': '태양계에서 가장 큰 행성은 무엇인가요?', 'test_input': '이 문장은 테스트문장입니다.', 'text': 'System: 이 시스템은 주어진 질문에 답변할 수 있습니다.\nHuman: 태양계에서 가장 큰 행성은 무엇인가요?, 이 문장은 테스트문장입니다.'}
```

