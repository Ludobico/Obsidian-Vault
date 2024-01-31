```python
RecursiveCharacterTextSplitter(separators: Optional[List[str]] = None, keep_separator: bool = True, is_separator_regex: bool = False, **kwargs: Any)
```

`RecursiveCharacterTextSplitter` 는 <font color="#ffff00">특정 문서를 특정 크기의 단위로 분할하는 역할</font>을 합니다.

RecursiveCharacterTextSplitter 는 다음과 같은 파라미터를 지정할 수 있습니다.

> chunk_size -> int
- 각 텍스트 조각의 최대 길이입니다. 영어는 글자당 1, 한국어는 글자당 2를 차지하며, 이 값이 작을수록 텍스트가 더 작은 조각으로 분할됩니다.

> chunk_overlap -> int
- 각 텍스트 조각이 겹치는 부분의 길이입니다. 이 값이 클수록 텍스트 조각 사이에 겹치는 부분이 더 많이 생기게됩니다. 여기서 겹치는 부분이란
	- overlap = 0일경우 **[안녕하세요, 반갑습니다]** 라는 문장이 overlap=2일경우 **[안녕하세요, 요반갑습니다]** 이런 형식으로 겹쳐집니다.

> length_function -> len()
- 문자열의 길이를 계산하는 함수입니다. 기본값은 `len()` 입니다.

> is_separator_regex -> bool
- 구분 기호를 정규 표현식으로 지정할지 여부입니다. 기본값은 `False` 로, 구분 기호를 문자로 지정합니다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])

```

```
page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and'
page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.'
```

```python
text_splitter.split_text(state_of_the_union)[:2]
```

```
['Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and',
 'of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.']
```

