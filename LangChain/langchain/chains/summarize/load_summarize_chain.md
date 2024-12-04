`load_summarize_chain` 은 [[LangChain/LangChain|LangChain]] 에서 **텍스트 요약**을 수행할 수 있도록 제공되는 Chain을 생성하는 함수입니다. 이 체인은 주어진 입력 텍스트를 요약하기 위해 다양한 summarization type 과 LLM을 결합하여 작동합니다.

```python
from langchain.chains import load_summarize_chain

# 예시: LLM을 사용하여 텍스트 요약
llm = OpenAI()
chain = load_summarize_chain(llm, chain_type="map_reduce")

# 텍스트 요약 실행
summary = chain.run(documents)
print(summary)

```

## Parameters

> llm -> BaseLanguageModel

- 텍스트 요약에 사용할 LLM을 지정합니다.

> chain_type -> str, default "map_reduce"

- Summarization type을 지정합니다.
	- `map_reduce` : 텍스트를 작은 조각으로 나누고 각각을 요약한 후 최종 요약을 생성합니다.
	- `stuff` : 전체 텍스트를 한 번에 요약합니다.
	- `refine` : 첫 번째 요약을 생성한 후, 여러 단계에 걸쳐 점진적으로 개선합니다.

> verbose -> bool, Default False


