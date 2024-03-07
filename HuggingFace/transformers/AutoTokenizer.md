- [[#AutoTokenizer.from_pretrained()|AutoTokenizer.from_pretrained()]]
- [[#Tokenizer Methods|Tokenizer Methods]]
- [[#Tokenizer Methods#encode|encode]]
- [[#Tokenizer Methods#decode|decode]]


AutoTokenizer 는 [[HuggingFace🤗]] 의 [[transformers]]라이브러리에서 제공하는 도구 중 하나로, 자연어 처리(NLP) 모델을 사용하기 전에<font color="#ffff00"> 텍스트 데이터를 모델이 이해할 수 있는 형식으로 변환</font>해주는 역할을 하는 클래스입니다.

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
- 파일 다운로드 중에 중단된 파일이 있는 경우 해당 파일을 삭제할지 여부를 나타내는 불리언 값입니다. 이 매개변수를 <font color="#ffc000">True</font>로 설정하면 중단된 파일이 있을 경우 다운로드를 재개하려고 시도합니다.

> proxies -> dict[str, str] , (optional)
- 프록시 서버를 지정하는데 사용되는 딕셔너리입니다. 프록시는 프로토콜 또는 엔드포인트 별로 정의될 수 있습니다. 예를 들어
```bash
{'http': 'foo.bar:3128', '[http://hostname](http://hostname/)': 'foo.bar:4012'}
```

와 같이 지정할 수 있습니다. 이를 통해 다운로드 요청 시 프록시 서버를 사용할 수 있습니다.

> revision -> str , (optional), defaults to "main"
- 사용할 모델 버전을 나타내는 문자열입니다. 이것은 <font color="#ffff00">브랜치 이름, 태그 이름 또는 커밋 ID</font> 가 될 수 있으며, huggingface.co의 모델과 기타 아티팩트를 저장하는데 Git 기반 시스템을 사용하므로 Git에서 허용하는 모든 식별자를 사용할 수 있습니다.

> subfolder -> str, (optional)
- 관련 파일이 위치한 서브폴더를 지정하는 문자열입니다. 모델 및 토크나이저와 관련된 파일이 특정 서브폴더에 저장되어 있을 때 사용합니다.

> use_fast -> bool, (optional), Defaults to True
- 지정된 모델이 지원하는 경우 해당 모델에 대해 <font color="#ffff00">빠른 Rust 기반 토크나이저를 사용할지 여부</font>를 나타내는 불리언 값입니다. 빠른 토크나이저가 지원되지 않는 경우 일반적인 Python 기반 토크나이저가 반환됩니다.

> tokenizer_type -> str, (optional)
- 로드할 토크나이저 타입입니다. 이 파라미터를 사용하면 특정 유형의 토크나이저를 명시적으로 선택할 수 있습니다. 예를 들어 `BertTokenizer` 나 `GPT2Tokenizer` 와 같은 토크나이저 유형을 지정할 수 있습니다. 이 파라미터를 생략하면 자동으로 적절한 토크나이저 유형이 선택됩니다.

> trust_remote_code -> bool, (optional), Defaults to False
- Hub에 정의된 사용자 정의 모델을 사용할 수 있도록 할지 여부입니다. 이 옵션은 신뢰성 있는 레파지토리에서만 사용해야 하며, 해당 레파지토리의 코드를 확인한 후에만 <font color="#de7802">True</font> 로 설정해야 합니다. 왜냐하면 이 옵션을 <font color="#de7802">True</font> 로 설정하면 Hub에 있는 코드가 로컬 머신에서 실행되기 때문입니다. 즉, <font color="#ffff00">외부에서 제공되는 코드를 신뢰할 수 있는지 확인하고 그에 따라 옵션을 설정</font>해야 합니다.

> kwargs -> additional keyword arguments, (optional)
- 추가적인 키워드 인자로 `Tokenizer__init__()` 메서드로 전달됩니다. 이를 사용하여 `bos_token` `eos_token` `unk_token` `sep_token` `pad_token` 과 같은 특수 토큰을 설정할 수 있습니다.


## Tokenizer Methods
---
`AutoTokenizer.from_pretrained()` 으로 [[Tokenizer]] 인스턴스화 시킬 수 있습니다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-v1.0")
```

인스턴스된 토크나이저는 다음과 같은 메서드를 사용할 수 있습니다.

### encode

이 encode 메서드는 <font color="#ffff00">주어진 입력을 정수 시퀀스로 변환하는 역할</font>을 합니다. 이 메서드는 [[Tokenizer]] 와 어휘 사전(vocab dict)를 사용하여 작동합니다. 주어진 텍스트를 정수 시퀀스로 변환하기 위해 먼저 tokenize 메서드를 사용하여 텍스트를 토큰화하고, `convert_tokens_to_ids` 메서드를 사용하여 토큰을 정수로 변환합니다.

주요 파라미터는 다음과 같습니다.

> text -> str, List[str] or List[int]

- 첫번째 시퀀스입니다. 문자열, 토큰화된 문자열의 리스트, 또는 정수의 리스트를 인자로 받습니다.

> text_pair -> str, List[str] or List[int], optional
- 두 번째 시퀀스입니다.

> add_special_tokens -> bool, optional
- True로 설정하면 시퀀스의 특수 토큰이 추가됩니다.

```python
  class TokenizerExam:
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-v1.0")
  @classmethod
  def special_token_test(cls):
    tokenizer = cls().tokenizer
    print(tokenizer.encode("I am learning how to use transformers for text generation.", add_special_tokens=False))
    print('-'*50)
    print(tokenizer.encode("I am learning how to use transformers for text generation.", add_special_tokens=True))
```

```
[315, 837, 5168, 910, 298, 938, 5516, 404, 354, 2245, 8342, 28723]
--------------------------------------------------
[1, 315, 837, 5168, 910, 298, 938, 5516, 404, 354, 2245, 8342, 28723]
```


> max_length -> int, optional
- 반환되는 시퀀스의 최대 길이를 제한하는 데 사용됩니다.

> stride -> int, optional
- `max_length` 와 함께 설정된 경우, 반환된 오버플로우 토큰에는 주요 시퀀스의 일부 토큰이 포함됩니다.

> truncation_strategy -> str, optional
- 주어진 옵션 중 하나를 선택하여 입력 시퀀스를 잘라내는 strategy를 지정합니다.

> pad_to_max_length -> bool, optional
- True로 설정하면 반환된 시퀀스가 모델의 최대 길이까지 [[padding]] 됩니다.

> return_tensors -> str, optional
- '<font color="#ffff00">tf' </font>또는 <font color="#ffff00">'pt'</font> 로 설정하여 TensorFlowtf.constant 또는 [[Pytorch]] 의 [[torch.Tensor]] 를 반환합니다.

### decode

이 decode 메서드는 <font color="#ffff00">정수 시퀀스를 문자열로 변환하는 역할</font>을 합니다. 이 메서드는 토크나이저와 어휘 사전을 사용하여 작동합니다. 주어진 정수 시퀀스를 토큰으로 변환하고 이를 문자열로 결합하여 반환합니다.

주요 파라미터는 다음과 같습니다.

> token_ids -> List[int]
- 토큰화된 입력 시퀀스의 정수 ID 리스트입니다. 이는 `encode` 또는 `encode_plus` 메서드를 사용하여 얻을 수 있습니다.

> skip_special_tokens -> bool, default by False
- True로 설정하면 특수 토큰을 제거하고 일반 토큰만을 포함한 문자열을 반환합니다.

> clean_up_tokenization_spaces -> bool, default by False
- True로 설정하면 토큰화 과정에서 추가된 공백을 제거하여 더 정확한 문자열을 생성합니다.

