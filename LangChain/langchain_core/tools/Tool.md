[[LangChain/LangChain|LangChain]] 의 `Tool` 클래스는 언어모델이 특정 작업을 수행하거나 외부 데이터를 처리할 수 있도록 함수나 코루틴을 감싸는 객체입니다. 이를 통해 **언어모델이 더 복잡한 작업을 처리** (예 : agent) 할 수 있는 인터페이스를 제공합니다.

## Parameters

> args_schema -> optional, BaseModel

- `args_schema` 는 Tool이 입력값을 받았을 때, 해당 값을 검증하고 파싱하기 위해 사용됩니다. 이 파라미터는 `Pydantic.BaseModel` 또는 `pydantic.v1.BaseModel` 의 서브클래스로 정의할 수 있습니다. 예를 들어, 입력값으로 이름과 나이를 받아야 한다면, 이를 `pydantic` 모델로 정의해 검증로직을 자동화 할 수 있습니다.

```python
from pydantic import BaseModel

class InputSchema(BaseModel):
    name: str
    age: int

```

- `Tool` 의 입력값이 이 모델과 일치하지 않으면 자동으로 `ValidationError` 가 발생하며, 이를 `handle_validation_error` 로 처리할 수 있습니다.

> callbacks -> optional, BaseCallbackHandler,

- `callbacks` 는 리스트 형태로 여러 개의 콜백 핸들러를 지정할 수 있으며, 각 핸들러는 tool 실행 중 특정 이벤트(예 : 실행 시작, 완료, 실패 등)에서 호출됩니다.

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_tool_start(self, tool_name, input_data):
        print(f"Tool {tool_name} started with input: {input_data}")

tool = Tool(
    func=lambda x: x,
    callbacks=[MyCallback()]
)

```

> coroutine -> optional, Callable, Awaitable

- `coroutine` 은 비동기 작업을 지원하기 위해 제공되는 파라미터입니다. [[Python]] 의 [[Asyncio]] 코루틴을 받아들여 비동기 API 호출, 파일 읽기/쓰기 등 시간이 오래 걸리는 작업을 효율적으로 처리할 수 있습니다.
- 코루틴을 사용하는 경우 비동기 함수는 `ainvoke` 또는 `arun` 메서드를 통해 호출됩니다.

```python
import asyncio

async def async_function(input_text):
    await asyncio.sleep(1)
    return f"Async processed: {input_text}"

tool = Tool(
    coroutine=async_function,
    description="Asynchronous tool example."
)

# 호출
result = asyncio.run(tool.arun("Test input"))
```

> description -> str

- `description` 은 Tool의 역할, 사용 방법, 목적을 설명합니다. 이 설명은 주로 AI 모델이 **해당 도를 언제, 어떻게 사용할지 결정하는 데 사용**됩니다. Few-shot 예제를 포함할 수도 있으며, 이를 통해 모델이 도구 사용법을 더 잘 이해할 수 있도록 도웁니다.

```python
tool = Tool(
    func=lambda x: x,
    description="This tool reverses the input string. Example: Input 'abc' -> Output 'cba'."
)

```

> func -> optional, Callable, str

- `func` 는 동기 작업을 처리하기 위해 사용되는 파라미터입니다. 호출 가능한 Python 함수로 정의되며, Tool이 실행될 때 이 함수가 호출됩니다. func는 입력값을 받아서 처리한 후 문자열 형태의 결과를 반환해야 합니다.

```python
def reverse_string(input_text):
    return input_text[::-1]

tool = Tool(
    func=reverse_string,
    description="Reverses the input text."
)

result = tool.run("hello")  # 출력: "olleh"
```

> handle_tool_error -> Optional, Union[bool, str, Callable]

- Tool 실행 중 예외(`ToolException`) 가 발생했을 때, 이를 처리하는 방법을 정의합니다.
	- `True` : 예외 메시지를 기본 처리 방식으로 반환
	- `str` : 예외가 발생했을 때 반환할 고정 메시지
	- `Callable` : 예외 객체를 입력으로 받아 커스텀 메시지를 반환하는 함수

```python
def error_handler(exception):
    return f"Error occurred: {exception}"

tool = Tool(
    func=lambda x: 1 / 0,
    handle_tool_error=error_handler
)

result = tool.run("Test input")  # 출력: "Error occurred: division by zero"
```

> handle_validation_error -> Optional, Union[bool, str, Callable]

- `args_schema` 실패 시 발생하는 `ValidationError` 를 처리합니다. 사용 방법은 `handle_tool_error` 와 유사합니다.

> metadata -> optional, Dict[str, any]

- `metadata` 는 Tool 호출과 관련된 추가 정보를 저장합니다. 이를 통해 Tool 실행 컨텍스트를 식별하거나, 호출 데이터를 추적할 수 있습니다. 메타데이터는 `callbacks` 에 정의된 핸들러로 전달되이를 활용해 추가적인 로깅이나 분석 작업을 수행할 수 있습니다.

```python
tool = Tool(
    func=lambda x: x,
    metadata={"tool_version": "1.0", "author": "Alice"}
)

```

> response_format -> Literal [content, content_and_artifact], Default : content

- Tool의 반환값 형식을 정의합니다.
	- "content" : Tool이 문자열 형태의 내용을 반환
	- "content_and_artifact" : `(내용, 추가 데이터)` 의 튜플 형태로 반환

> return_direct -> bool, Default : False

- `return_direct` 가 `True` 로 설정되면, Tool 실행 후 결과값이 즉시 반환되며, 에이전트 실행 루프 중단됩니다. 이 기능은 특정 Task에서 **Tool의 결과를 최종 출력**으로 삼고자 할 때 사용합니다.

```python
tool = Tool(
    func=lambda x: x.upper(),
    return_direct=True
)

result = tool.run("hello")  # 출력: "HELLO"

```

> tags -> optional, List[str]

- `tags` 는 Tool 호출 시 특정 태그를 지정하여 분류하거나 추적하는데 사용합니다.

> verbose -> bool, Default : False

- `verbose` 는 Tool 실행 과정을 로깅할지 여부를 제어합니다.

