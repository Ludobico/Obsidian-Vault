FewShotPromptTemplate 은 [[LangChain/LangChain|LangChain]] 의 중요한 기능 중 하나입니다. 이는 자연어 처리 모델에 **몇 가지 예시와 함께 프롬프트를 제공하여 모델이 해당 작업을 효과적으로 학습** 할 수 있도록 돕는 기술입니다.

Few-Shot 학습은 많은 양의 데이터 없이도 모델이 새로운 작업을 수행할 수 있도록 합니다. 이를 위해 해당 작업에 대한 몇 가지 예시를 모델에 제공하고, 이를 기반으로 모델이 일반화된 규칙을 파악하도록 합니다.

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
```

아래는 파라미터에 대한 설명입니다.

> example_prompt -> [[PromptTemplate]] , required

- 이 파라미터는 각각의 예제를 어떻게 형식화할지를 정의하는 [[PromptTemplate]] 입니다. 이는 필수 요소로, 각각의 개별 예제를 어떻게 표시할지를 결정합니다.

> example_selector -> Optional[BaseExampleSelector]

- `BaseExampleSelector` 를 활용하여 사용할 예제를 선택하는 파라미터입니다. 이 파라미터나 `examples` 중 하나가 제공되어야 합니다.

> example_separator -> str = '\n\n'

- 예제들 사이, 그리고 프롬프트의 접두사(prefix)와 점미사(suffix) 사이에 사용될 문자열 구분자입니다. 기본값은 두 개의 줄바꿈입니다.

 > examples -> Optional, List[dict]
- 직접 리스트 형태로 제공된 예제들입니다. 이 파라미터나 `example_selector` 중 하나가 제공되어야 합니다.

> input_types -> Dict[str, Any], optional

- [[PromptTemplate]] 이 기대하는 변수의 유형을 명시하는 딕셔너리입니다. 제공됮 않을 경우 **몯ㄴ 변수는 문자열로 간주** 됩니다.

> input_variables -> List[str], required

- [[PromptTemplate]] 에 필요한 변수들의 이름을 나열한 것입니다. 