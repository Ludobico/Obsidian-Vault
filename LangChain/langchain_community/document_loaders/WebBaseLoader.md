
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path = "https://www.espn.com/"
    # header_template = None,
    # verify_ssl = True,
    # proxies = None,
    # continue_on_failure = False,
    # autoset_encoding = True,
    # encoding = None,
    # web_paths = (),
    # requests_per_second = 2,
    # default_parser = "html.parser",
    # requests_kwargs = None,
    # raise_for_status = False,
    # bs_get_text_kwargs = None,
    # bs_kwargs = None,
    # session = None,
    # show_progress = True,
)
```

`WebBaseLoader` 는 **웹 페이지에서 데이터를 가져오는 기능**을 제공하는 [[LangChain/LangChain|LangChain]] 의 로더입니다. 이 로더는 bs4를 사용하여 HTML을 파싱하고, 여러 웹 페이지에서 데이터를 가져오는 작업을 자동화합니다. 각 파라미터는 다양한 설정을 제어하며, 웹 데이터를 로드하는 방식에 영향을 미칩니다.

## Example

```python
from langchain_community.document_loaders import WebBaseLoader

def load_document(url):
    """
    Load document from the specified URL
    """
    loader = WebBaseLoader(url)
    return loader.load()
    

if __name__ == "__main__":
    result = load_document("https://example.com")
    print(result[0].page_content)
```

```python
docs = []
docs_lazy = loader.lazy_load()

# async variant:
# docs_lazy = await loader.alazy_load()

for doc in docs_lazy:
    docs.append(doc)
print(docs[0].page_content[:100])
print(docs[0].metadata)
```


## Parameters

> web_paths -> Sequence[str]
- 로드할 웹 URL을 나열하는 문자열 시퀀스입니다.
- \["https://example.com", "https://another-example.com"]

> requests_per_second -> int
- 초당 요청 수를 제한하는 값입니다. 너무 많은 요청을 동시에 보내지 않도록 조절하여 서버의 부하를 방지합니다.

> default_parser -> str
- bs4에서 사용하는 기본 파서(parser) 입니다. `html.parser` , `lxml`, `html5lib` 등 다양한 파서를 선택할 수 있습니다.

> requests_kwargs -> optional, Dict[str, Any]
- `requests` 라이브러리의 요청 시 추가적인 인자를 설정할 수 있는 딕셔너리입니다. 예를 들어 `headers` , `timeout` 등을 설정할 수 있습니다.
- {"timeout": 10, "headers": {"User-Agent": "my-agent"}}

> raise_for_status -> bool
- HTTP 요청에서 오류 상태 코드 (404, 500 등)를 받았을 때 예외를 발생시킬지 여부를 설정합니다. `True` 이면 오류 발생시 예외를 발생시킵니다.

> bs_get_text_kwargs -> optional, Dict[str, Any]
- bs4의 `get_text()` 메서드를 호출할 때 추가로 전달할 수 있는 인자들입니다. 예를 들어, `separator` 나 `strip` 옵션을 설정할 수 있습니다.

> bs_kwargs -> optional, Dict[str, Any]
- bs4로 웹 페이지를 파싱할 때 사용할 추가적인 설정들입니다. 예를 들어, 특정 파서나 옵션을 지정할 수 있습니다.

> show_progress -> bool
- 웹 페이지 로딩 진행 상황을 나타내는 progress bar를 표시할지 여부입니다.

> web_path -> Union[str, Sequence[str]]
- 로드할 웹 경로로, 문자열 또는 문자열 시퀀스를 받습니다. `web_paths` 와 유사하지만, 한 번에 하나의 URL 만 로드하려면 문자열을 사용할 수 있습니다.

> header_template -> optional, dict
- 요청 시 사용할 HTTP 헤더의 템플릿입니다. 여러 요청에서 동일한 헤더를 사용할 때 유용합니다.

> verify_ssl -> bool
- SSL 인증서 검증 여부를 설정합니다. `True` 로 설정하면 인증서를 검증하고, `False` 로 설정하면 인증서 오류를 무시합니다.

> proxies -> optional, dict
- 요청에 사용할 프록시 서버를 설정할 수 있는 딕셔너리입니다. 주로 네트워크 제한이 있을 때 사용합니다.

> continue_on_faulure -> bool
- 웹 요청 실패 시 계속 진행할지 여부입니다. `True` 로 설정하면 실패한 요청을 무시하고 계속 다음 페이지를 로드합니다.

> autoset_encoding -> bool
- 자동으로 페이지의 인코딩을 설정할지 여부입니다. `True` 면 웹 페이지의 `charset` 을 자동으로 추론하여 인코딩을 설정합니다.

> encoding -> optional, str
- 요청 시 사용할 인코딩을 수동으로 설정할 수 있습니다. 일반적으로 `utf-8` 이 사용됩니다.

> session -> Any
- `requests.Session()` 객체를 설정하여 세션을 공유할 수 있습니다. 이 객체를 사용하면 요청 간 세션 정보를 유지하고, 쿠키나 헤더를 공유할 수 있습니다.

