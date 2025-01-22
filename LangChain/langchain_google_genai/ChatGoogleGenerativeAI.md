`ChatGoogleGenerativeAI` 는 **Google의 생성형 AI(Chat model)** 을 통합하여 [[LangChain/LangChain|LangChain]] 과 같은 라이브러리에서 Google AI의 기능을 활용할 수 있도록 설계되었습니다.

## Instantiation

`ChatGoogleGenerativeAI` 를 사용하려면 **Google API key** 가 필요합니다. 이를 설정하는 방법은 두 가지가 있습니다.

### Set the API key as environment variable

```bash
export GOOGLE_API_KEY=your_google_api_key_here
```


### Provide the API Key through keyword argument

```python
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(google_api_key="your_google_api_key_here")
```


## Parameters

