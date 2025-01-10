
`RunnableConfig` 는 [[LangChain/LangChain|LangChain]] 의 [[runnables]] 를 설정하기 위한 구성 클래스입니다. 이를 통해 실행 중에 **태그, 메타데이터, 콜백 및 기타 실행 환경** 속성을 정의할 수 있습니다.

각각의 속성은 실행 중의 tracing, data management와 관련된 기능을 제공합니다. 아래는 각 속성에 대한 설명입니다.

## Parameters

> tags -> List\[str\]

- [[runnables]] 와 모든 하위(체인이 LLM을 호출할 경우)를 위한 tag list 입니다.

> metadata -> Dict\[str, any\]

- 메타데이터를 정의합니다. 주로 <font color="#ffff00">시작 또는 완료 이벤트에서 컨텍스트 정보를 전달</font>하는데 사용됩니다.

> callbakcs -> Optional\[Union\[List, Any\]\]

- 콜백을 정의합니다. 실행 상태 추적, 로깅, 또는 특정 이벤트 발생 시 동작을 정의하는 데 사용됩니다.

> run_name -> str

- 이 호출에 대한 이름입니다. 기본값은 클래스의 이름으로 설정됩니다.

> max_concurrency -> Optional\[int\]

- 병렬로 실행할 수 있는 최대 호출 수를 지정합니다.
- 동시 실행 제한을 통해 리소스 사용량을 관리합니다.

> recursion_limit -> int

- 호출이 재귀적(recursive)으로 호출할 수 있는 최대 수를 지정합니다.
- 기본값은 25입니다.

> configurable -> Dict\[str, Any\]

- 실행 중에 구성 가능한 속성값을 지정합니다.

