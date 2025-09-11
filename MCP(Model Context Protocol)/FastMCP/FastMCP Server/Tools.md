- [[#Tools|Tools]]
- [[#What are tools?|What are tools?]]
- [[#The @tool Decorator|The @tool Decorator]]
- [[#Decorator Arguments|Decorator Arguments]]
	- [[#Decorator Arguments#parameters|parameters]]

## Tools

[[MCP(Model Context Protocol)]] 의 클라이언트가 호출할 수 있는 함수들의 목록입니다.

도구(Tools)는 LLM이 외부 시스템과 상호작용하고, 코드를 실행하며, 학습 데이터에 포함되지 않은 데이터에 접근할 수 있게 해주는 핵심 컴포넌트입니다. [[FastMCP]] 에서는 일반적인 [[Python]] 함수를 MCP 프로토콜을 통해 LLM이 사용할 수 있는 도구로 제작할 수 있습니다.

## What are tools?

FastMCP에서 Tool은 평범한 파이썬 함수를 LLM이 직접 호출할 수 있는 실행 가능한 기능으로 변환합니다.
이를 통해 LLM은 DB 조회, API 호출, 계산, 파일 접근 등 원래 모델이 알 수 없는 작업까지 수행할 수 있습니다.

## The @tool Decorator

Tool을 만드는 방법은 간단합니다. 파이썬 함수에 `@mcp.tool` 데코레이터를 붙이면 됩니다.

```python
from fastmcp import FastMCP

mcp = FastMCP(name="CalculatorServer")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b
```

이 함수가 Tool로 등록되면 FastMCP가 자동으로 처리해줍니다.

함수 이름(`add`) 이 Tool 이름으로 사용되고, 함수의 Docstring이 Tool의 description으로 사용되는 것은 [[LangChain/LangChain|LangChain]] 의 [[LangChain/langchain_core/tools/tools|tools]] 와 비슷합니다.

## Decorator Arguments

FastMCP는 기본적으로 함수의 이름과 docstring을 사용해 Tool 이름과 description을 생성합니다. 하지만 `@mcp.tool` 데코레이터에 인자를 추가하면 이름과 설명을 직접 지정하고, 태그나 메타데이터 같은 부가 정보도 붙일 수 있습니다.

```python
@mcp.tool(
    name="find_products",  # LLM에 노출될 Tool 이름
    description="카테고리별로 필터링 가능한 상품 검색 기능",  # LLM에 노출될 설명
    tags={"catalog", "search"},  # 정리/필터링용 태그
    meta={"version": "1.2", "author": "product-team"}  # 커스텀 메타데이터
)
def search_products_implementation(query: str, category: str | None = None) -> list[dict]:
    """내부용 함수 설명 (위 description이 지정되면 무시됨)."""
    print(f"'{category}' 카테고리에서 '{query}' 검색 중…")
    return [{"id": 2, "name": "Another Product"}]

```

### parameters

> name -> str

- MCP에 사용될 Tool 이름을 명시적으로 지정합니다.

> description -> str

- MCP에 사용될 설명입니다. tool안에 지정되면 함수의 docstring은 무시됩니다.

> tags -> str[str]

- Tool을 분류하기 위한 태그의 집합입니다. 서버나 클라이언트에서 Tool을 필터링하거나 그룹화할 때 활용이 가능합니다.

> enabled -> bool, default : True

- Tool 활성화 여부입니다. 비활성화 옵션은 Disbling Tools 문서에 있습니다.

> exclude_args -> list[str]

- LLM에 사용되는 Tool 스키마에서 특정 인자를 제외할 수 있습니다.

> annotations -> dict

- Tool에 추가 메타데이터를 붙일 수 있는 딕셔너리입니다.

> meta -> dict[str, any]

- Tool에 커스텀 메타 정보를 추가할 수 있습니다.

