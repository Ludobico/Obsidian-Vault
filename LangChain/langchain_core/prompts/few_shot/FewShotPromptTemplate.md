FewShotPromptTemplate 은 [[LangChain/LangChain|LangChain]] 의 중요한 기능 중 하나입니다. 이는 자연어 처리 모델에 **몇 가지 예시와 함께 프롬프트를 제공하여 모델이 해당 작업을 효과적으로 학습** 할 수 있도록 돕는 기술입니다.

Few-Shot 학습은 많은 양의 데이터 없이도 모델이 새로운 작업을 수행할 수 있도록 합니다. 이를 위해 해당 작업에 대한 몇 가지 예시를 모델에 제공하고, 이를 기반으로 모델이 일반화된 규칙을 파악하도록 합니다.

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
```

아래는 파라미터에 대한 설명입니다.

> example_prompt -> [[PromptTemplate]] , **required**

- 이 파라미터는 각각의 예제를 어떻게 형식화할지를 정의하는 [[PromptTemplate]] 입니다. 이는 필수 요소로, 각각의 개별 예제를 어떻게 표시할지를 결정합니다.

> example_selector -> Optional[BaseExampleSelector]

- `BaseExampleSelector` 를 활용하여 사용할 예제를 선택하는 파라미터입니다. 이 파라미터나 `examples` 중 하나가 제공되어야 합니다.

> example_separator -> str = '\n\n'

- 예제들 사이, 그리고 프롬프트의 접두사(prefix)와 점미사(suffix) 사이에 사용될 문자열 구분자입니다. 기본값은 두 개의 줄바꿈입니다.

 > examples -> Optional, List[dict]
- 직접 리스트 형태로 제공된 예제들입니다. 이 파라미터나 `example_selector` 중 하나가 제공되어야 합니다.

> input_types -> Dict[str, Any], optional

- [[PromptTemplate]] 이 기대하는 변수의 유형을 명시하는 딕셔너리입니다. 제공됮 않을 경우 **몯ㄴ 변수는 문자열로 간주** 됩니다.

> input_variables -> List[str], **required**

- [[PromptTemplate]] 에 필요한 변수들의 이름을 나열한 것입니다. 

> metadata -> Dict[str, Any], optional

- 트레이싱을 위해 사용될 수 있는 메타데이터입니다.

> output_parser -> BaseOutputParser, optional

- LLM에 의해 형성된 프롬프트의 출력을 어떻게 파싱할지를 정의하는 `BaseOutputParser` 입니다.

> partial_variables -> Mapping[str, Any], optional

- 템플릿을 호출할 때마다 전달할 필요가 없는 변수들의 딕셔너리입니다.

> prefix -> str = ''

- 예제들 앞에 높일 프롬프트 템플릿 문자열입니다. 기본값은 빈 문자열("") 입니다.

> suffix -> str , **required**

- 예제들 뒤에 놓일 프롬프트 템플릿 문자열입니다.

> tags -> list[str], optional

- 트레이싱을 위해 사용될 태그 리스트입니다.

> template_format -> Literal['f-string', 'jinja2'] = 'f-string'

- 프롬프트 템플릿의 형식을 결정합니다. 옵션은 'f-string' 또는 'jinja2' 입니다. 이는 템플릿의 구문및 처리 방식에 영향을 미칩니다.

> validate_template -> bool, defaults to False

- 템플릿이 유효한지 검증 시도 여부를 결정하는 불리언 값입니다.

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]
```

```python
example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

print(example_prompt.format(**examples[0]))
```


```python
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

