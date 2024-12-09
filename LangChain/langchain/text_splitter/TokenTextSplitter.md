- [[#Parameters|Parameters]]
- [[#Methods|Methods]]
	- [[#Methods#atransform_documents()|atransform_documents()]]
	- [[#Methods#create_documents()|create_documents()]]
	- [[#Methods#from_huggingface_tokenizer()|from_huggingface_tokenizer()]]
	- [[#Methods#from_tiktoken_encoder()|from_tiktoken_encoder()]]
	- [[#Methods#split_documents()|split_documents()]]
	- [[#Methods#split_text()|split_text()]]
	- [[#Methods#transform_documents()|transform_documents()]]


`TokenTextSplitter` 는 **텍스트를 토큰 단위로 분할** 하는 [[LangChain/LangChain|LangChain]] 의 [[text_splitter]] 클래스입니다.

주로 LLM의 토크나이저를 활용하여 토큰 개수 기반으로 텍스트를 나누는 데 사용됩니다.

```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=200,  # 청크 크기를 10으로 설정합니다.
    chunk_overlap=0,  # 청크 간 중복을 0으로 설정합니다.
)

# state_of_the_union 텍스트를 청크로 분할합니다.
texts = text_splitter.split_text(file)
print(texts[0])  # 분할된 텍스트의 첫 번째 청크를 출력합니다.
```


## Parameters

> encoding_name -> str, Default 'gpt2'

- 사용할 토크나이저의 인코딩 이름을 지정합니다.

> model_name -> optional, str

- 특정 언어 모델을 기반으로 토큰화를 수행합니다. `gpt-4` 나 `gpt-3.5-turbo` 와 같은 모델 이름을 사용할 수 있습니다.

> allowd_special -> Union\[Literal\['all'], AbstactSet\[str]]

- 허용할 특수 토큰의 집합을 지정합니다. 시스템 메시지나 특정 토큰을 유지하려는 경우 사용합니다.

> disallowed_special 

- 허용하지 않는 특수 토큰을 지정합니다. 기본값은 `all` 로 모든 특수 토큰을 허용하지 않습니다.



## Methods

### atransform_documents()

주어진 문서 리스트를 비동기적으로 변환합니다.

> documents -> Sequence\[Document\]
- 변환해야 할 문서 리스트

> kwargs


### create_documents()

텍스트와 메타데이터를 사용해 `Document` 객체의 리스트를 생성합니다.

> text -> List\[str\]
- 텍스트 리스트

> medatatas -> optional, List\[dict\]
- 각 텍스트에 대응하는 메타데이터 리스트

### from_huggingface_tokenizer()

[[HuggingFace🤗]] 의 토크나이저를 사용하여 텍스트 길이를 계산하는 Text spliiter를 생성합니다.

> tokenizer -> Any
- HuggingFace [[Tokenizer]] 객체

> kwargs

### from_tiktoken_encoder()

Tiktoken 인코더를 사용하여 테스트 길이를 계산하는 Text splitter를 생성합니다.

> encoding_name -> str
- 사용하려는 인코딩의 이름 (예 : gpt2)

> model_name -> optional, str
- 모델 이름

> allowed_special -> Union\[Literal\['all'\]\, AbstractSet\[str\]\]
- 허용할 특수 토큰

> disallowed_special -> Union\[Literal\['all'\]\, AbstractSet\[str\]\]
- 허용되지 않는 특수 토큰

> kwargs

```python
text_splitter = TokenTextSplitter.from_tiktoken_encoder(encoding_name="gpt2")
```
### split_documents()

여러 문서를 분할합니다.

> documents -> Iteralble[Document]
- 분할하려는 문서의 iterable
### split_text()

텍스트를 여러 요소로 분할합니다.

> text -> str
- 분할하려는 텍스트

```python
split_texts = token_text_splitter.split_text("This is a sample text.")
```
### transform_documents()

주어진 문서들을 분할한 뒤 변환합니다.

> documents -> Sequence\[Document\]
- 변환 대상 문서의 시퀀스

> kwargs

```python
transformed_docs = token_text_splitter.transform_documents(documents)
```

