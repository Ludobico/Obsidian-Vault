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
	- <font color="#ffff00">디렉토리 경로</font> : 필요한 토크나이저의 어휘 파일이 포함된 디렉토리 경로를 지정할 수 있습니다.
	- <font color="#ffff00">단어 어휘 파일 경로</font> : 토크나이저가 단일 어휘 파일만 필요한 경우 해당 파일의 경로를 지정할 수 있습니다.

> inputs -> (optional)
- [[Tokenizer]] 의 `__init__()` 메서드에 전달할 추가 인수입니다. Tokenizer의 생성자에 전달할 인수를 지정하는데 사용됩니다.

> config -> (optional)
- 선택적으로 사용할 PretrainedConfig 객체입니다. 이를 통해 어떤 종류의 토크나이저를 인스턴스화할지 결정할 수 있습니다.

> cache_dir -> str or os.PathLike (optional)
- 다운로드된 사전 학습된 모델 설정을 캐시할 디렉토리의 경로입니다. 표준 캐시를 사용하지 않도록 설정할 경우 이 경로를 사용합니다.

> force_download -> bool, (optional), Defaults to False
- 모델 가중치 및 설정 파일을 강제 다운로드하도록 여부를 나타내는 불리언 값입니다. 캐시된 버전을 무시하고 다운로드를 강제하려면 <font color="#ffc000">True</font>로 설정합니다.

> resume_download -> bool, (optional) , Defaults to False
- 파일 다운로드 중에 중단된 파일이 있는 경우 해당 파일을 삭제할지 여부를 나타내는 불리언 값입니다. 이 매개변수를 True로 설정하면 중단된 파일이 있을 경우 다운로드를 재개하려고 시도합니다.





