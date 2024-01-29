`TextLoder` 는 [[LangChain]] 에서 제공하는 문서 로더인 TextLoder를 가져오는 문법입니다. TextLoader는 <font color="#ffff00">단순한 텍스트 파일을 읽어 하나의 문서로 만드는 역할</font>을 합니다.

```python
from langchain_community.document_loaders import TextLoader
import os

class VectorStore:
  @staticmethod
  def text_split():
    PATH = os.path.join(os.getcwd(), 'backend', 'Utils', 'wiki.txt')
    loader = TextLoader(PATH, encoding='utf-8')
    document = loader.load()
    print(document)


if __name__ == "__main__":
  VectorStore.text_split()

```

```
[
    Document(page_content='---\nsidebar_position: 0\n---\n# Document loaders\n\nUse document loaders to load data from a source as `Document`\'s. A `Document` is a piece of text\nand associated metadata. For example, there are document loaders for loading a simple `.txt` file, for loading the text\ncontents of any web page, or even for loading a transcript of a YouTube video.\n\nEvery document loader exposes two methods:\n1. "Load": load documents from the configured source\n2. "Load and split": load documents from the configured source and split them using the passed in text splitter\n\nThey optionally implement:\n\n3. "Lazy load": load documents into memory lazily\n', metadata={'source': '../docs/docs/modules/data_connection/document_loaders/index.md'})
]
```


