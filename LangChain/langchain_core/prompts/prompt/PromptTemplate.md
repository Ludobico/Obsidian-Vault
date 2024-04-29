
`PromptTemplate` 은 [[LangChain/LangChain|LangChain]] 라이브러리의 일부로, **언어 모델을 위한 프롬프트 템플릿을 생성하고 관리하는 기능을 제공**합니다. 이 클래스를 사용하면 사용자가 입력 변수를 기반으로 텍스트 템플릿을 정의하고, 이를 통해 언어 모델에 전달할 특정 문구를 동적으로 생성할 수 있습니다.

## example code

```python
from langchain_core.prompts import PromptTemplate

# Instantiation using from_template (recommended)
prompt = PromptTemplate.from_template("Say {foo}")
prompt.format(foo="bar")

# Instantiation using initializer
prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
```

> input_types -> Dict[str, Any], optional
- 템플릿에서 사용될 변수들의 데이터 타입을 정의하는 딕셔너리입니다. 여기서 각 변수의 이름을 키로, 해당 변수의 데이터 타입을 값으로 설정합니다.
- 데이터 타입이 명시되지 않은 경우, 모든 변수는 문자열로 간주됩니다.

> input_variables -> List[str]
- 프롬프트 템플릿에 필요한 변수들의 이름을 나열하는 리스트입니다. 이 리스트에 명시된 모든 변수는 템플릿을 사용할 때 제공되어야 합니다.

> metadata -> Dict[str, Any], optional
- 템프릿과 관련된 메타데이터를 설정할 수 있는 딕셔너리입니다. 이 정보는 로깅,추적 또는 기타 관리 목적으로 사용될 수 있습니다.

> output_parser -> BaseOutputParser, optional
- 프롬프트를 언어 모델에 전달한 후 얻은 출력을 어떻게 파싱할지 정의하는 파서 객체입니다. 이 파서는 출력 데이터의 처리 방법을 지정합니다.

> partial_varialbles -> Mapping[str, Any], optional
- 프롬프트 템플릿에서 반복적으로 사용되는 변수의 값을 딕셔너리로 설정할 수 있습니다. 이렇게 설정된 벼수들은 프롬프트 호출 시마다 제공할 필요가 없으며, 템플릿에서 자동으로 포함됩니다.

> tags -> List[str], optional
- 템플릿에 태그를 추가할 수 있으며, 이 태그들은 추적이나 분류에 사용될 수 있습니다.

> template -> str
- 프롬프트 템플릿의 실제 텍스트를 정의하는 문자열입니다. 이 템플릿 문자열 내에서 중괄호 `{}` 를 사용하여 변수를 포함시킬 수 있습니다.

> template_format -> Literal\['f-string', 'mustache', 'jinja2\'] = 'f-string'
- 템플릿의 형식을 지정하는 파라미터로, `f-string`, `mustache`, `jinja2` 중에서 선택할 수 있습니다. 기본값은 `f-string` 입니다.

> validate_template -> bool
- 템플릿의 유효성을 검사할지 여부를 결정하는 불리언 값입니다. 이를 `True` 로 설정하면, 템플릿이 유요한 형식인지 검사합니다.

