`RunnableLambda`  는 [[LangChain/LangChain|LangChain]]에서  **사용자 정의 함수를 체인에 통합** 할 수 있게 해주는 클래스입니다. 일반 [[Python]] 함수를 Langchain의 Runnable 형태로 변환하여 다른 체인 컴포넌트들과 함께 사용할 수 있게 해줍니다.

```python
from config.getenv import GetEnv
from Utils.highlight import highlight_print
from Module.prompts import PromptList

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

env = GetEnv()
api_key = env.get_openai_api_key

#  Runnable
from langchain_core.runnables import  RunnableLambda

def add_prefix(text : str) -> str:
    return f"처리된 텍스트 : {text}"

processor = RunnableLambda(add_prefix)

result = processor.invoke("안녕하세요.")
highlight_print(result)

model = ChatOpenAI(temperature=0, api_key=api_key, model='gpt-4o-mini')

template = "{text}에 대해 설명해주세요"
prompt = PromptTemplate(input_variables=['text'], template=template)

chain = (
    prompt | RunnableLambda(add_prefix) | model | StrOutputParser()
)

result = chain.invoke("안녕하세요.")

highlight_print(result)
```

```
--------------------------------------------------------------------------------
처리된 텍스트 : 안녕하세요.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
안녕하세요! "안녕하세요"는 한국어에서 인사할 때 사용하는 표현입니다. 일반적으로 처음 만났을 때나 누군가에게 인사를 할 때 사용됩니다. 이 표현은 상대방에게 친근감을 주고 대화를 시작하는 좋은 방법입니다. 추가로 궁금한 점이나 다른 주제에 대해 설명이 필요하시면 말씀해 주세요!
--------------------------------------------------------------------------------
```

