- [[#Parameters|Parameters]]
- [[#example code ( using method )|example code ( using method )]]
- [[#deep dive to ChatPromptTemplate|deep dive to ChatPromptTemplate]]


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

[[PromptTemplate]] 과 다르게 프롬프트를 구성할때 `template` 파라미터가 아닌 `messages` 파라미터를 받습니다.

```python
messages = [
    ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."),
    ("human", "안녕하세요, 오늘의 날씨는 어떤가요?"),
    ("ai", "안녕하세요! 제가 실시간 날씨 정보를 제공하지는 못하지만, 일반적인 날씨 정보에 대해 답변드릴 수 있습니다. 특정 지역의 날씨를 알고 싶으시다면 말씀해 주세요."),
    ("human", "{location}의 {season} 날씨는 어떤가요?")
]
chat_prompt = ChatPromptTemplate(messages=messages, input_variables = ['location', 'season'])

chat_prompt.pretty_print()
```

```
================================ System Message ================================

당신은 친절하고 도움이 되는 AI 어시스턴트입니다.

================================ Human Message =================================

안녕하세요, 오늘의 날씨는 어떤가요?

================================== AI Message ==================================

안녕하세요! 제가 실시간 날씨 정보를 제공하지는 못하지만, 일반적인 날씨 정보에 대해 답변드릴 수 있습니다. 특정 지역의 날씨를 알고 싶으시다면 말씀해 주세요.

================================ Human Message =================================

{location}의 {season} 날씨는 어떤가요?
```

## Parameters

대부분의 파라미터는 [[PromptTemplate]] 의 파라미터와 공유합니다.

> messages

- 메시지의 시퀀스입니다. Chat Prompt의 구성 요소로, 각 메시지를 다양한 형식으로 표현할 수 있습니다.
	- `BaseMessagePromptTemplate` 메시지를 템플릿으로 정의
	- [[BaseMessage]] 고정된 메시지 객체
	- 튜플 형태(메시지 타입 , 템플릿)
	- 튜플 형태(메시지 클래스, 템플릿)
	- 문자열

> template_format -> default : "f-string"

- 템플릿의 포맷 방식입니다. 기본값은 "f-string"([[Python]] 의 포맷팅 스타일) 입니다.

> input_variables

- 프롬프트 템플릿을 채우는 데 필요한 변수들의 이름 리스트입니다.

> partial_variables

- 미리 정의된 변수들로, 프롬프트 생성 시 매번 제공할 필요 없이 고정된 값을 갖습니다.

> optional_variable

- 옵션으로 제공할 수 있는 변수들의 이름 리스트로, **값이 제공되지 않으면 기본값으로 비워**둡니다.

> validate_template -> default : True

- 템플릿의 유효성을 검증할지 여부를 설정합니다.
- 템플릿 내 변수들의 정의와 `input_variables`, `partial_variables` 가 올바르게 매핑되었는지 확인합니다.

> input_types -> default : str

- 각 변수의 데이터 타입을 정의합니다.
- 모든 변수는 기본값으로 문자열로 간주합니다.
	- `input_types={"age": int, "name": str}`로 변수의 기대 타입을 명시할 수 있습니다.
## example code ( using method )

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

## deep dive to ChatPromptTemplate

```python
system_template = """
Your name is {model_name}, created by {owner_name}. 
The current date is {date}. 
{model_name}'s knowledge was last updated on {last_data_update}, and it provides answers to questions as an informed individual from that date would, while letting the user know when relevant. 

{model_name} offers concise responses for simple queries and clarifies when tasks exceed its capabilities (e.g., opening URLs or videos). When asked to present views held by many people, {model_name} provides balanced information, even if it personally disagrees, while also presenting broader perspectives. It avoids stereotyping, particularly negative stereotyping of any group, and addresses controversial topics with objective and careful analysis, without downplaying harmful content or implying there are “reasonable” perspectives in all cases.

When a response contains highly specific details about rare or obscure topics that are unlikely to be widely available online, {model_name} may end its response with a reminder that it could "hallucinate," a term the user understands. This caution is omitted when information is widely known, even if the topic is somewhat obscure. 

{model_name} is skilled at tasks such as writing, analysis, coding, math, and more, using markdown for code and providing accurate, natural-sounding responses in Korean.
"""

human_template = """
{question}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template),
]

prompt = ChatPromptTemplate(messages=messages, input_variables=['question'], partial_variables={"model_name" : model_name, "owner_name" : owner_name, "date" : date_time, "last_data_update" : last_data_update})

prompt.pretty_print()
```

```
================================ System Message ================================


Your name is {model_name}, created by {owner_name}.
The current date is {date}.
{model_name}'s knowledge was last updated on {last_data_update}, and it provides answers to questions as an informed individual from that date would, while letting the user know when relevant.

{model_name} offers concise responses for simple queries and clarifies when tasks exceed its capabilities (e.g., opening URLs or videos). When asked to present views held by many people, {model_name} provides balanced information, even if it personally disagrees, while also presenting broader perspectives. It avoids stereotyping, particularly negative stereotyping of any group, and addresses controversial topics with objective and careful analysis, without downplaying harmful content or implying there are “reasonable” perspectives in all cases.

When a response contains highly specific details about rare or obscure topics that are unlikely to be widely available online, {model_name} may end its response with a reminder that it could "hallucinate," a term the user understands. This caution is omitted when information is widely known, even if the topic is somewhat obscure.

{model_name} is skilled at tasks such as writing, analysis, coding, math, and more, using markdown for code and providing accurate, natural-sounding responses in Korean. 


================================ Human Message =================================


{question}
```

## Methods

- async
> abatch()
> 	- input -> List[input]
> 	- config -> optional, RunnableConfig = None
> 	- return_exceptions -> bool, Default : False
> 	- \*\*kwargs

[[Asyncio]]의 gather를 사용해서 병렬로 ainoke()를 실행하는 기본 메서드입니다.
이 메서드의 각 배치는 입출력(IO) 중심의 실행 가능 작업과 호환됩니다.

- async
> abatch_as_completed()
> 	- inputs -> Sequence[input]
> 	- config -> optional, RunnableConfig
> 	- return_exceptions -> bool, Default : False
> 	- \*\*kwargs

병렬로 리스트 input값을 받고 ainovke()를 실행하는 메서드입니다.  작업이 완료되는 대로 결과를 반환합니다.

- async
> aformat()
> 	- \*\*kwargs

chat template을 문자열 형식으로 비동기적으로 포맷합니다.


- async
> aformat_messages()
> 	- \*\*kwargs

chat template을 메시지 형식으로 비동기적으로 포맷합니다.

- async
> aformat_prompt()
> 	- \*\*kwargs

비동기적으로 프롬프트를 포맷하고 [[PromptValue]] 를 반환합니다.

- async
> ainvoke
> 	- input -> Dict
> 	- config -> RunnableConfig
> 	- \*\*kwargs

비동기적으로 프롬프트를 호출합니다.

-  append
> 	- message : Union[[[BaseMessage]]]

chat template 의 끝부분에 메시지를 추가합니다.

