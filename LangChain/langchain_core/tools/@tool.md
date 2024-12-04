`@tool` 데코레이터는 [[LangChain/LangChain|LangChain]] 에서 **커스텀 툴(Custom tool)** 을 정의하는 방법을 제공합니다. 이 데코레이터를 사용하면 <font color="#ffff00">함수의 이름</font>과 <font color="#00b050">Docstring</font>을 자동으로 <font color="#ffff00">툴의 이름</font>과 <font color="#00b050">설명</font>으로 설정하여 에이전트가 해당 툴을 사용할 수 있도록 만듭니다.

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

print(search.name)
print(search.description)
print(search.args)
```

```
search
search(query: str) -> str - Look up things online.
{'query': {'title': 'Query', 'type': 'string'}}
```

```python
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(multiply.name)  
print(multiply.description)  
print(multiply.args)
```

```
multiply
multiply(a: int, b: int) -> int - Multiply two numbers.
{'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
```

또한 `@tool` 데코레이터의 파라미터를 설정하여 tool name 과 스키마를 정의할 수 있습니다.

```python
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")


@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

print(search.name)  
print(search.description)  
print(search.args)  
print(search.return_direct)
```

```
search-tool  
search-tool(query: str) -> str - Look up things online.  
{'query': {'title': 'Query', 'description': 'should be a search query', 'type': 'string'}}  
True
```

