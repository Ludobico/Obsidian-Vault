- [[#Structure of the Document Object|Structure of the Document Object]]
- [[#RAG Pipeline|RAG Pipeline]]
	- [[#RAG Pipeline#Document Loaders|Document Loaders]]
	- [[#RAG Pipeline#Text Splitting|Text Splitting]]
	- [[#RAG Pipeline#Embedding & VectorStore|Embedding & VectorStore]]
	- [[#RAG Pipeline#Retrieval|Retrieval]]


![[Pasted image 20250716101422.png]]

```python
from langchain_core.documents import Document

document = Document(
    page_content="Hello, world!",
    metadata={"source": "https://example.com"}
)
```

[[LangChain/LangChain|LangChain]] 의 `Document` 객체는 RAG(Retrieval-Augmented Generation) 워크플로우에서 핵심적인 데이터 구조로, **텍스트 데이터와 그에 대한 메타데이터를 체계적으로 관리하여 검색 및 generation 작업을 지원**합니다.

## Structure of the Document Object

LangChain의 `Document` 객체는 두 가지 주요 속성으로 구성됩니다.

- <font color="#ffff00">page_content</font> (문자열)
	- 문서의 실제 텍스트 콘텐츠를 포함합니다.
	- RAG 파이프라인에서 [[embedding]] 되어 벡터스토어에 저장되며, 주로 검색(retrieval) 과 LLM의 입력으로 사용됩니다.
	- 예 : PDF 문서의 특정 페이지 텍스트, 웹페이지 본문, CSV 파일의 레코드 등

- <font color="#ffff00">metadata</font> (딕셔너리)
	- 문서에 대한 추가 정보를 키-값 쌍으로 저장합니다.
	- 검색 필터링, 문서 추적, 컨텍스트 강화 등에 사용됩니다.
	- 예 : `{"source": "doc1.pdf", "page": 5, "author": "John Doe", "date": "2023-10-01"}`


`Document` 객체는 langchain의 RAG 워크플로우에서 데이터를 구조화하고 관리하는 기본 단위입니다. 주요역할을 다음과 같습니다.

- 데이터 저장 : page_content 와 metadata를 하나의 객체로 통합
- 검색 지원 : page_content 는 벡터 검색의 대상이 되고, metadata는 검색 범위를 좁히거나 결과를 정제는 데 사용
- 컨텍스트 제공 : LLM에 전달될 때 page_content와 metadata 를 함께 활용해 더 정확하고 맥락에 맞는 답변 생성
- 출처 추적 : metadata를 통해 문서의 출처(예 : 파일 이름, URL, 페이지 번호) 를 명시하여 신뢰성과 투명성 확보

## RAG Pipeline

RAG 파이프라인은 일반적으로<font color="#ffff00"> 문서 로딩 -> 청킹 -> 임베딩 -> 벡터스토어 저장 -> 검색 -> 생성</font> 단계로 구성되며, `Document` 객체는 각 단계에서 핵심적인 역할을 합니다. 아래는 단계별로 어떻게 사용되는지 설명합니다.

### Document Loaders

- 역할 : 다양한 소스(pdf, csv, txt, json 등)에서 데이터를 로드해 `Document` 객체로 변환
- page_content : 소스에서 추출된 텍스트 데이터
- metadata : 소스 정보(파일 경로, url), 페이지 번호, 행 번호, 작성자 등 자동 또는 수동으로 추가

```python
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path="mlb_teams_2012.csv")
docs = loader.load()
```

### Text Splitting

- 역할 : 긴 문서를 작은 조각(chunk)으로 나누어 검색 효율성을 높임
- page_content : 각 청크는 원본 문서의 일부 텍스트를 포함
- metadata : 원본 문서의 메타데이터를 상속하거나, 청크별 추가 정보(청크 id, 부모 문서 id 등) 추가

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
# 출력: 각 청크는 새로운 Document 객체로, metadata는 원본 문서에서 상속됨
```

### Embedding & VectorStore

- 역할 : page_conent 를 임베딩하여 벡터로 변환하고, 벡터스토어에 저장
- page_content : 벡터화의 대상
- metadata : 벡터와 함께 저장되어 검색 시 필터링에 사용

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
```

### Retrieval

- 역할 : 사용자의 쿼리에 따라 벡터스토어에서 관련 문서를 검색
- page_content : 쿼리와의 유사도를 계산해 관련성 높은 문서 선택
- metadata : 검색 전 필터링(출처나 날짜) 또는 결과 정제에 사용

```python
results = vectorstore.similarity_search(
    query="LangChain이란?",
    k=3,
    filter={"source": "langchain_docs.pdf"}
)
```

