`RunnableParallel` 은 [[LangChain/LangChain|LangChain]] 에서 **여러 작업을 동시에 실행하고 그 결과들을 하나의 딕셔너리로 결합**하는 클래스입니다.

`RunnableParallel` 은 하나의 입력을 받아 여러 작업을 병렬로 처리하고, 각 작업의 결과를 지정된 키와 함께 딕셔너리 형태로 반환합니다. 주로 다음과 같은 상황에서 사용됩니다.

- 하나의 입력에 대해 여러 가지 독립적인 처리가 필요한 경우

- 여러 작업의 결과를 구조화된 형태로 관리해야 하는 경우

- 처리 시간을 단축하기 위해 작업을 병렬화해야 하는 경우

```python
import os, sys, pdb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from config.getenv import GetEnv
from Utils.highlight import highlight_print
from Module.prompts import PromptList

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

env = GetEnv()
api_key = env.get_openai_api_key

#  Runnable
from langchain_core.runnables import  RunnableParallel

model = ChatOpenAI(temperature=0, api_key=api_key, model='gpt-4o-mini')

# 두 개의 다른 프롬프트 템플릿 생성

summary_template = """
다음 텍스트를 한 문장으로 요약해주세요:
{text}
"""
summary_prompt = PromptTemplate(input_variables=['text'], template=summary_template)

translation_template = """
다음 텍스트를 영어로 번역해주세요:
{text}
"""
translation_prompt = PromptTemplate(input_variables=['text'], template=translation_template)


summary_chain = summary_prompt | model | StrOutputParser()
translation_chain = translation_prompt | model | StrOutputParser()

parallel_chain = RunnableParallel(
    summary = summary_chain,
    translation = translation_chain,
)

question = """
오늘은 날씨가 좋습니다. 아침에 일찍 일어나 빠르게 샤워를 하고
공원에서 산책을 했습니다.
"""
result = parallel_chain.invoke({'text' : question})

highlight_print(result['summary'])
highlight_print(result['translation'])
```

```
--------------------------------------------------------------------------------
오늘 날씨가 좋고, 아침에 일찍 일어나 샤워 후 공원에서 산책했습니다.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
Today the weather is nice. I woke up early in the morning, took a quick shower, and went for a walk in the park.
--------------------------------------------------------------------------------
```

