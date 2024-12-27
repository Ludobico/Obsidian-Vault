- [[#Parameters|Parameters]]
- [[#example|example]]


`trim_messages` [[LangChain/LangChain|LangChain]] 에서  **주어진 메시지 시퀀스를 주어진 token limit 이하로 줄이기 위해 사용**됩니다.

## Parameters

> messages

- 자를 메시지의 리스트입니다. 

> max_tokens

- 자른 후 메시지가 포함할 수 있는 최대 토큰 수

> token_counter

- 토큰 수를 계산하는 함수나 객체입니다.
- Langchain의 `BaseLanguageModel` 을 전달하면, 모델의 `get_num_tokens_from_messages()` 메서드를 사용합니다.

> strategy -> default : "last"

- 메시지를 자르는 전략을 설정합니다.
- 기본값은 `last` 이며, 마지막 메시지로부터 토큰 제한 수에 맞게 자릅니다.

> allow_partial -> default : False

- 부분적으로 메시지를 잘라 포함할지 여부를 설정.
- 기본값은 `False` 입니다.

> end_on

- 특정 메시지 타입 이후의 메시지를 무시합니다.
- 예를 들어 "human" 으로 설정하면 [[HumanMessage]] 이후의 메시지가 제거됩니다.

> start_on

- 특정 메시지 타입 이전의 메시지를 무시합니다.
- `strategy` 파라미터가 `last` 일 경우만 사용가능합니다.

> include_system

- 처음에 있는 [[SystemMessage]] 를 유지할지 여부를 설정합니다.

> text_splitter

- 메시지 텍스트를 분리하는 함수 또는 [[text_splitter]] 객체를 받습니다.


## example

```python
from langchain_core.messages import trim_messages, AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("why is 42 always the answer?"),
    AIMessage(
        "Because it’s the only number that’s constantly right, even when it doesn’t add up!"
    ),
    HumanMessage("What did the cow say?"),
]

trimmed_messages = trim_messages(
    messages = messages,
    token_counter = len,
    max_tokens = 5,
    strategy = "last",
    allow_partial=True,
    include_system=True
)

for msg in trimmed_messages:
    msg.pretty_print()
```

```
================================ System Message ================================

you're a good assistant, you always respond with a joke.
================================== Ai Message ==================================

Hmmm let me think.

Why, he's probably chasing after the last cup of coffee in the office!
================================ Human Message =================================

why is 42 always the answer?
================================== Ai Message ==================================

Because it’s the only number that’s constantly right, even when it doesn’t add up!
================================ Human Message =================================

What did the cow say?
```

