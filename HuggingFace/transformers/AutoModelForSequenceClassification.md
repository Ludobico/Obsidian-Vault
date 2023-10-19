AutoModelForSequenceClassification 는 [[transformers]] 라이브러리의 일부로, <font color="#ffff00">시퀀스 분류 작업을 수행하는 모델을 제어하는 제너릭 모델 클래스</font>입니다. 이 클래스는 `from_pretrained()` 클래스나 `from_config()` 클래스 메서드를 사용하여 생성될 때, 라이브러리의 모델 클래스 중 하나로 인스턴스화됩니다.

이 클래스는 직접 `__init__()` 메서드를 사용하여 인스턴스화할 수 없으며, 이렇게 시도하면 오류가 발생합니다. 대신 `from_pretrained()` 메서드 또는 `from_config()` 메서드를 사용하면 사전 학습된 모델을 로드하거나 구성 파일을 기반으로 새 모델을 만들 수 있습니다.

AutoModelForSequenceClassification 는 <font color="#ffff00">텍스트 시퀀스를 입력으로 받아, 주어진 시퀀스를 분류하는 모델을 나타내며</font>, 다양한 NLP 작업에 활용됩ㄴ다. 이 모델은 라이브러리의 사전 학습 모델 중 하나로 선택될 수 있고, 텍스트 분류, 감정 분석, 텍스트 유사성 평가 등과 같은 다양한 시퀀스를 수행하는 모델로 개발할 수 있습니다.

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
```

## <font color="#0070c0">AutoModelForSequenceClassification.from_pretrained()</font>
---
> pretrained_model_name_or_path -> str or os.PathLike
- 이 파라미터는 미리 학습된 모델의 이름 또는 경로를 나타냅니다. 이것은 다음과 같은 방식으로 정의될 수 있습니다.
	- 모델 레파지토리(huggingface.co) 에서 호스팅되는 미리 학습된 모델의 모델 ID. 이러한 모델 ID는 root-level에 위치하며 예를 들어 "<font color="#ffc000">bert-base-uncased</font>" 와 같은 이름을 가질 수 있습니다. 또는 user 및 organization 이름 아래에 네임스페이스화된 모델 ID도 가능합니다. 예를 들어 "<font color="#ffc000">dbmdz/bert-base-german-cased</font>" 와 같은 이름을 가질 수 있습니다.
	
	- 모델 가중치가 저장된 디렉토리의 경로. 이 경우, 모델 가중치를 `save_pretrained()` 메서드를 사용하여 저장한 디렉토리의 경로를 제공합니다. 예를 들어 "<font color="#ffc000">./my_model_directory/</font>"와 같은 경로입니다.
	
	- Pytorch state_dict 저장 파일의 경로 또는 URL. 이 경우 `from_pt` 를 True로 설정하고 config 객체를 인수로 제공해야합니다. 이 방식은 [[Pytorch]] 모델을 직접 로드하며 다른 방법보다 느립니다. 저장된 모델 가중치의 파일 경로를 제공합니다. 예를 들어 "<font color="#ffc000">./pt_model/pytorch_model.bin</font>"과 같은 경로입니다.

> model_args -> additional positional arguments, (optional)
- 이 파라미터는 모델의 생성자(`__init__()`) 로 전달되는 추가 매개변수입니다. 모델 생성자에서 필요한 경우에 사용됩니다.

> config -> PretrainedConfig, (optional)
- 모델의 구성을 나타내는 `PretrainedConfig` 객체입니다. 이 파라미터를 사용하여 모델에 대한 구성을 명시적으로 제공할 수 있으며, 자동으로 로드되는 구성 대신 사용됩니다. 구성은 다음 상황에서 자동으로 로드됩니다.
	- 모델이 라이브러리에서 제공되는 모델인 경우
	- 모델이 `save_pretrained()` 를 사용하여 저장되었고, 저장 디렉토리를 제공하여 다시 로드되는 경우
	- 모델이 로컬 디렉토리를 `pretrained_model_name_or_path` 로 제공하고 디렉토리 내에 <font color="#ffc000">config.json</font> 이라는 config JSON 파일이 있는 경우

> cache_dir -> str or os.PathLike, (optional)
- 다운로드한 미리 학습된 모델 구성을 캐시할 디렉토리의 경로를 지정합니다. 기본적으로는 표준 캐시를 사용하지만, 특정 디렉토리에 구성을 캐시하도록 지정할 수 있습니다.

> from_pt -> bool, (optional), Defaults to False
- [[Pytorch]] 체크포인트 저장 파일에서 모델 가중치를 로드할지 여부를 나타냅니다. 이 매개변수를 <font color="#ffc000">True</font>로 설정하면 Pytorch 체크포인트 파일에서 모델 가중치를 로드합니다.

> force_download -> bool, (optional), Defaults to False
- 모델 가중치와 config 파일을 다시 다운로드할지 여부를 나타냅니다. 기본적으로 캐시된 파일이 있으면 캐시된 버전을 사용하지만, <font color="#ffc000">True</font>로 설정하면 캐시된 버전을 무시하고 모델 파일을 다시 다운로드합니다.

> resume_download -> bool, (optional), Defaults to False
- 파일 다운로드 중에 중단된 파일이 있는 경우, 이 파라미터를 <font color="#ffc000">True</font>로 설정하면 중단된 다운로드 파일을 삭제하고 중단된 다운로드를 재개합니다.

> proxies -> Dict[str, str], (optional)
- 프록시 서버를 지정하는 딕셔너리입니다. 각 프로토콜 또는 엔드포인트별로 프록시 서버를 지정할 수 있습니다. 예를 들어
```python
{'http': 'foo.bar:3128', '[http://hostname](http://hostname/)': 'foo.bar:4012'}
```

와 같이 사용할 프록시 서버를 지정할 수 있습니다.

> output_loading_info -> bool, (optional), Defaults to False
- <font color="#ffc000">True</font>로 설정하면 로딩 과정에서 누락된 키, 예상치 못한 키 오류 메시지를 포함하는 딕셔너리를 반환합니다. 로딩 정보를 확인하려면, 이 파라미터를 True로 설정합니다.

> local_files_only -> bool, (optional), Defaults to False
- True로 설정하면 로컬 파일만 검톻고 모델을 다운로드하지 않습니다. 이 옵션을 사용하여 로컬 파일에서 모델을 로드하거나 다운로드를 시도하지 않도록 설정할 수 있습니다.

> revision -> str, (optional), Defaults to "main"
- 모델의 특정 버전을 지정하는데 사용됩니다. Git 기반의 시스템을 사용하여 모델 및 기타 자산을 저장하기 때문에 이 파라미터는 git에서 허용되는 식별자(브랜치 이름, 태그 이름 또는 커밋 ID)로 설정할 수 있습니다.

> trust_remote_code -> bool, (optional), defaults to False
- True로 설정하면 허브의 자체 모델링 파일에서 사용자 정의 모델을 허용합니다. 그러나 이 옵션을 True로 설정하는 경우, <font color="#ffff00">코드를 읽은 후에만 신뢰할 수 있는 레파지토리에서 사용해야 합니다</font>. 허브에서 제공된 코드를 로컬 머신에서 실행하므로 주의해야 합니다.

> code_revision -> str, (optional), Defaults to "main"
- 모델 코드가 모델의 나머지 부분과 다른 레포지토리에 있는 경우 코드의 특정 리비전을 지정하는데 사용됩니다. Git 기반의 시스템을 사용하므로 브랜치 이름, 태그 이름 또는 커밋 ID와 같은 식별자로 설정됩니다.

> kwargs -> additional keyword arguments, (optional)
- config 객체가 자동으로 로드되거나 수동으로 제공되는 경우에 동작이 다릅니다. 이 파라미터를 사용하여 config 객체를 업데이트하거나 모델을 초기화할 수 있습니다. 예를 들어, `output_attentions = True`와 같이 키워드 인수를 사용하여 구성을 업데이트할 수 있습니다.







