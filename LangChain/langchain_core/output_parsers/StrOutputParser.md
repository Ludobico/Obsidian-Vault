
## Langchain and StrOutputParser
---

[[LangChain/LangChain|LangChain]] 은 LLM을 활용한 애플리케이션의 개발, 제품화, 배포를 용이하게 하게 포괄적인 프레임워크입니다. 이 프레임워크의 핵심 구성 요소 중 하나는 **다양한 출력 파서(output parser)** 입니다. 그 중 <font color="#ffff00">StrOutputParser</font> 는 그 단순함과 효과성으로 주목받고 있습니다.

## StrOutputParser
---
`StrOutputParser`는 **LLM 또는 ChatModel의 출력을 문자열 형식으로 변환** 하는 기능이 있습니다. 이 기능은 추가 처리나 최종 사용자에게 정보를 표시하기 위해 일관된 출력 형식을 필요로 하는 애플리케이션에서 매우 중요합니다.

## example code
---

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
chain.invoke({"input": "how can langsmith help with testing?"})
```

