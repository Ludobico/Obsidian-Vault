`MessagesPlaceholder` 는 주로 **채팅 기록(chat history)** 을 저장하고 관리하는데 사용되는 컴포넌트입니다.

핵심 기능은 유연한 메시지 삽입으로 주요 특징으로는 **대화 이력 저장, 에이전트의 중간 추론 과정(scratchpad) 저장, 동적으로 메시지 리스트를 프롬프트에 삽입** 등이 있습니다.


```python
# 대화 이력 저장
chat_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history")
])

# 에이전트 스크래치패드 저장 
agent_prompt = ChatPromptTemplate.from_messages([ MessagesPlaceholder(variable_name="agent_scratchpad")
])

```

### Message Placeholder with buffer memory

```python
import os, sys, pdb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

from config.getenv import GetEnv

env = GetEnv()
apikey = env.get_openai_api_key

llm = ChatOpenAI(temperature=0.1, api_key=apikey, model='gpt-4o-mini')

parser = StrOutputParser()

messages = [
    ("system", "당신은 유용하고 친절한 AI 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = (
    {
        "question": RunnablePassthrough(),
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
    }
    | prompt 
    | llm
    | parser
)

response = chain.invoke("나는 한국에 살고있다.")
print(response)
memory.save_context({"input" : "나는 한국에 살고있다."}, {"output" : response})

response2 = chain.invoke("내가 어느나라에 살고 있지?")
print(response2)
```

```
안녕하세요! 한국에 살고 계시군요. 한국에 대해 궁금한 점이나 도움이 필요한 부분이 있다면 언제든지 말씀해 주세요. 어떤 이야기를 나누고 싶으신가요?

당신은 한국에 살고 있다고 말씀하셨습니다. 한국에 대해 더 알고 싶거나 다른 질문이 있으시면 언제든지 말씀해 주세요
```

