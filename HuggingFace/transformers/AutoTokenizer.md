AutoTokenizer 는 [[HuggingFace🤗]] 의 Transformers 라이브러리에서 제공하는 도구 중 하나로, 자연어 처리(NLP) 모델을 사용하기 전에<font color="#ffff00"> 텍스트 데이터를 모델이 이해할 수 있는 형식으로 변환</font>해주는 역할을 하는 클래스입니다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```


### AutoTokenizer.from_pretrained()
---
AutoTokenizer.from_pretrained() 은 미리 학습된 모델의 이름 또는 경로를 기반으로 해당 모델에 대한 토크나이저를 로드하는데 사용됩니다. 이 메서드는 모델 이름 또는 경로를 입력으로 받고 해당 모데에 맞는 토크나이저를 인스턴스화 합니다.

주요 파라미터로는

> pretrained_model_name_or_path -> str or os.PathLike
- 사용할 사전 학습된 모델의 이름 또는 경로를 나타내는 문자열 또는 경로입니다. 이 매개변수는 다음과 같은 방식으로 지정할 수 있습니다.
	- <font color="#ffff00">문자열</font> : [[HuggingFace🤗]] 모델 허브에서 호스팅되는 미리 정의된 토크나이저의 모델 id입니다. 모델 id는 huggingface.co 의 모델 레파지토리에서 찾을 수 있습니다.
	- 디렉토리 경로 : 필요한 토크나이저의 어휘 파일이 포함된 디렉토리 경로를 지정할 수 있습니다.




