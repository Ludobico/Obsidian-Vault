`create_retriever_tool` 은 [[LangChain/LangChain|LangChain]] 에서 **retriever를 tool로 변환** 해주는 편의 함수입니다. 이를 통해 Agent가 retrieval 작업을 수행할 수 있도록 지원합니다.

## Parameters

> retriever -> Baseretriever

- 사용할 Retriever 객체입니다.
- 예를 드어, FAISS 기반의 `retriever` `ElasticSearchRetriever` `PineconeRetriever` 등을 사용할 수 있습니다.

> name -> str

- tool의 이름입니다.
- 에이전트가 이 이름을 통해 retriever를 호출하므로, 고유하고 독립적인 이름으로 설정해야합니.

> description -> str

- tool의 설명입니다. 에이전트에게 retriever 의 기능을 명확히 설명하며, LLM이 이를 적절히 호출할 수 있도록 돕습니다.
- 예 `A tool for searching documents in the knowledge base`

> document_prompt -> optional, BasePromptTemplate

- 검색 결과(문서)를 사용할 때의 출력 형식을 지정하는 프롬프트 템플릿입니다. 기본적으로 지정하지 않으며, 검색 결과를 그대로 반환합니다.

> document_separator -> str

- 여러 개의 문서가 반환될 경우, 문서 간에 삽입되는 구분자입니다. 기본값은 개행 문자 `\n\n` 입니다.
- 예를 들어 `---` 으로 설정하면 문서들이 `---` 으로 구분됩니다.



## Example code

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import create_retriever_tool

# FAISS 기반 retriever 생성
embedding_model = OpenAIEmbeddings()
documents = ["Document 1", "Document 2", "Document 3"]
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_texts(documents)
vectorstore = FAISS.from_texts(split_docs, embedding_model)
retriever = vectorstore.as_retriever()

# Tool 생성
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="document_search",
    description="Search for relevant documents in the knowledge base.",
    document_separator="\n---\n"
)

# Tool 사용 예시
query = "What is the content of Document 1?"
results = retriever_tool.func(query)
print(results)
```

