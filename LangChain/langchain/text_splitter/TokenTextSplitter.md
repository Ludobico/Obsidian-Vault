`TokenTextSplitter` 는 **텍스트를 토큰 단위로 분할** 하는 [[LangChain/LangChain|LangChain]] 의 [[text_splitter]] 클래스입니다.

주로 LLM의 토크나이저를 활용하여 토큰 개수 기반으로 텍스트를 나누는 데 사용됩니다.

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

### create_documents()

### from_huggingface_tokenizer()

### from_tiktoken_encoder()

### split_documents()

### split_text()

### transform_documents()

