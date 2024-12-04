`BaseTool` 은 [[LangChain/LangChain|LangChain]] 에서 [[Tool]] 을 구현하기 위한 기본 클래스입니다.

이는 LangChain의 **에이전트 시스템이 외부 기능이나 서비스를 호출**할 수 있도록 도와주는 <font color="#ffff00">인터페이스</font> 역할을 합니다.

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class CustomCalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        return self._run(a, b, run_manager=run_manager.get_sync())
```
## Parameters

> args_schema -> optional, Type[BaseModel] , Default None

- Tool의 입력 인자를 검증하고 파싱하기 위해 사용되는 pydanic 모델을 지정합니다.

> callbacks -> optional, union[list[BaseCallbackHandler], BaseCallbackManager], Default None

- Tool 실행 중에 호출할 콜백 핸들러를 지정합니다. 여러 콜백 핸들러를 리스트로 전달하거나 단일 콜백 핸들러를 전달할 수 있습니다.

> description -> str

- Tool의 **설명을 제공하는 파라미터**입니다. LLM에게 이 도구를 어떻게, 언제, 왜 사용할지 알려주는 역할을 합니다.

- `description` 에 [[few_shot]] 예시를 포함시킬 수 있습니다.

> handle_tool_error -> optional, union\[bool, str, Callable\[\[ToolException], str]], Default False

- Tool 실행 중 발생한 ToolException을 처리하는 방법을 지정합니다.
- `True` : 오류를 처리하도록 설정
- `False` : 오류를 처리하지 않도록 설정
- `str` : 오류 메시지 포맷을 지정
- `Callable[[ToolException], str]` : 오류를 처리할 커스텀 함수를 지정

> handle_validation_error : optional, union\[bool, str, Callable\[\[ValidationError], str]], Default False

- Tool 실행 중 발생한 ValidationError을 처리하는 방법을 지정합니다.
- `True` : 오류를 처리하도록 설정
- `False` : 오류를 처리하지 않도록 설정
- `str` : 오류 메시지 포맷을 지정
- `Callable[[ToolException], str]` : 오류를 처리할 커스텀 함수를 지정

> metadata -> optional, Dict\[str, Any], Default None

- Tool과 관련된 메타데이터를 지정할 수 있습니다. 이 메타데이터는 Tool이 호출될 때마다 함께 전달되며, 콜백 핸들러에 인자로 전달됩니다.

- 이는 특정 Tool 인스턴스의 용도나 사용 사례를 식별하는 데 사용됩니다.

> response_format -> Literal\['content', 'content_and_artifact'], Default content

- Tool이 반환하는 response 형식을 지정합니다.
- `content` : ToolMessage 의 내용으로 출력됩니다.
- `content_and_artifact` : `(content, artifact)` 형식의 두 개의 값으로 결과를 반환합니다.

> return_direct -> bool, Default False

- Tool의 출력을 직접 반환할지 여부를 설정합니다.
- `True` 로 설정하면 Tool 호출 후 [[AgentExecutor]] 가 루프를 멈추고 즉시 결과를 반환합니다.

> tags -> optional, List\[str], Default None

- Tool에 태그를 추가할 수 있습니다. 이 태그는 콜백 핸들러의 인자로 전달되며, 특정 인스턴스를식별하거나 용도별로 분류할 수 있습니다.

> verbose -> bool, Default False

- Tool의 진행을 로그로 출력할지 여부를 설정합니다.

