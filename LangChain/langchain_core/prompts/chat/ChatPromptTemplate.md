`ChatPromptTemplate` 은 [[LangChain/LangChain|LangChain]] 에서 **대화형 프롬프트를 생성하고 관리**하는데 사용되는 클래스입니다.

각 메시지는 `튜플(tuple)` 형식으로 표현되며, 다음과 같은 구조를 가집니다.

```python
(role, content)
```

> role
- 메시지의 역할을 나타내는 문자열입니다.

> content
- 실제 메시지 내용입니다.

`ChatPromptTemplate` 에서 사용되는 주요 역할은 다음과 같습니다.

> system
- 대화의 전반적인 맥락이나 AI의 행동 지침을 설정합니다.

- 주로 대화 시작 시 한 번 사용됩니다.

> human
- 사용자가 AI에게 전달하는 질문이나 명령을 나타냅니다.

> ai
- AI가 생성한 응답 메시지입니다.

## example code

```python
from langchain_core.prompts import ChatPromptTemplate

messages = [
    ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."),
    ("human", "안녕하세요, 오늘의 날씨는 어떤가요?"),
    ("ai", "안녕하세요! 제가 실시간 날씨 정보를 제공하지는 못하지만, 일반적인 날씨 정보에 대해 답변드릴 수 있습니다. 특정 지역의 날씨를 알고 싶으시다면 말씀해 주세요."),
    ("human", "{location}의 {season} 날씨는 어떤가요?")
]

chat_prompt = ChatPromptTemplate.from_messages(messages).format_messages(location = "서울", season = "봄")

print(chat_prompt)

```

```
[SystemMessage(content='당신은 친절하고 도움이 되는 AI 어시스턴트입니다.', additional_kwargs={}, response_metadata={}), HumanMessage(content='안녕하세요, 오늘의 날씨는 어떤가요?', additional_kwargs={}, response_metadata={}), AIMessage(content='안녕하세요! 제가 실시간 날씨 정보를 제공하지는 못하지만, 일반적인 날씨 정보에 대해 답변드릴 수 
있습니다. 특정 지역의 날씨를 알고 싶으시다면 말씀해 주세요.', additional_kwargs={}, response_metadata={}), HumanMessage(content='서울의 봄 날씨는 어떤가요?', additional_kwargs={}, response_metadata={})]
```

생성한 메시지를 LLM과 연결하여 다음과 같은 결과를 출력할 수 있습니다.

```python
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config.getenv import GetEnv

env = GetEnv()
apikey = env.get_openai_api_key

prompt = [
    ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."),
    ("human", "안녕하세요, 오늘의 날씨는 어떤가요?"),
    ("ai", "안녕하세요! 제가 실시간 날씨 정보를 제공하지는 못하지만, 일반적인 날씨 정보에 대해 답변드릴 수 있습니다. 특정 지역의 날씨를 알고 싶으시다면 말씀해 주세요."),
    ("human", "{location}의 {season} 날씨는 어떤가요?")
]
chat_prompt = ChatPromptTemplate.from_messages(prompt)

llm = ChatOpenAI(temperature=0.1, api_key=apikey, model='gpt-4o-mini')

parser = StrOutputParser()

chain = chat_prompt | llm | parser

print(chain.invoke({"location" : "대한민국", "season" : "봄"}))
```

```
대한민국의 봄 날씨는 대체로 온화하고 쾌적합니다. 보통 3월부터 5월까지 이어지며, 다음과 같은 특징이 있습니다:

1. **3월**: 겨울의 추위가 점차 풀리기 시작하지만, 여전히 쌀쌀한 날씨가 많습니다. 평균 기온은 5도에서 15도 사이로 변동합니다. 이 시기에는 꽃이 피기 시작하며, 특히 벚꽃이 
유명합니다.

2. **4월**: 기온이 더 따뜻해지며, 평균 기온은 10도에서 20도 사이입니다. 벚꽃이 만개하고, 다양한 꽃들이 피어나는 시기입니다. 날씨가 맑고 화창한 날이 많아 야외 활동에 적합
합니다.

3. **5월**: 봄의 마지막 달로, 기온이 더욱 올라 평균 15도에서 25도 사이입니다. 날씨가 매우 쾌적하고, 많은 사람들이 야외 활동을 즐깁니다. 이 시기에는 초여름의 기운도 느껴지기 시작합니다.

봄철에는 비가 오는 날도 있지만, 대체로 맑고 화창한 날이 많아 여행이나 소풍에 적합한 계절입니다.
```

