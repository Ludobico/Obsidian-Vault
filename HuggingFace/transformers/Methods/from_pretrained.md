<font color="#ffff00">from_pretrained() </font> 메소드는 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 사전 훈련된 모델을 불러오는 데 사용됩니다. 이 메소드는 다양한 매개변수를 통해 모델의 불러오기 및 설정을 맞춤화할 수 있습니다. 아래는 <font color="#ffff00">from_pretrained() </font> 메소드의 주요 매개변수들에 대한 설명입니다.

```python
from transformers import AutoConfig, AutoModelForCausalLM

# Download model and configuration from huggingface.co and cache.
model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
model.config.output_attentions

# Loading from a TF checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForCausalLM.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)
```

> pretrained_model_name_or_path -> str or os.PathLike

- 이 매개변수는 사전 훈련된 모델의 식별자 또는 모델이 저장된 경로를 지정합니다. 사용할 수 있는 옵션은 다음과 같습니다.
	- [[HuggingFace🤗]] 의 모델 저장소에 호스티왼 사전 훈련된 모델의 ID
	- [[save_pretrained]] 를 통해 저장된 디렉토리 경로

> model_args -> additional arguments, optional

- 이 인수들은 내부 모델의 `__init__()` 메소드로 전달됩니다.

> config -> [[PretrainedConfig]] , optional

- 모델의 구성을 지정합니다. **구성은 자동으로 로드되지만 사용자가 직접 지정할 수도 있습니다.** 모델이 라이브러리에 의해 지정되거나 [[save_pretrained]] 를 통해 저장된 경우 구성이 자동으로 로드됩니다.

> state_dict -> Dict[str, [[torch.Tensor]] ] , optional

- 사전 훈련된 모델의 가중치 대신 사용할 state_dict를 지정합니다. 이 옵션은 사전 훈련된 구성으로 모델을 생성하되 **사용자 자신이 가중치를 로드하고자 할 때 유용**합니다.

> cache_dir -> str or os.PathLike , optional

- 사전 훈련된 모델 구성을 **캐싱할 디렉토리 경로를 지정** 합니다. 기본 캐시를 사용하지 않으려는 경우에 사용됩니다. 

> from_tf -> bool, optional, defaults to False

- TensorFlow 체크포인트 파일로부터 모델 가중치를 로드할지 여부를 결정합니다. TensorFlow 체크포인트를 사용하는 경우 이 값을 `True` 로 설정해야 합니다.

> force_download -> bool, optional, defaults to False

- 모델의 가중치와 구성 파일을 캐시된 버전을 무시하고 강제로 다시 다운로드할지 여부를 결정합니다. 이미 캐시된 파일이 존재하는 경우에도 새로 다운로드하고 싶을 때 사용합니다.

> resume_download -> bool, optional, defaults to False

- 불완전하게 수신된 파일을 삭제할지의 여부를 결정하고, 해당 파일이 존재하는 경우 다운로드를 재개합니다. **네트워크 중단이나 다운로드 실패 후 이어서 다운로드하고자 할 때 유용** 합니다.

> proxies -> Dict[str, str], optional

- 프로토콜 또는 엔드포인트 별로 사용할 프록시 서버의 사전입니다. 예를 들어
```python
{'http': 'foo.bar:3128','http://hostname': 'foo.bar:4012'}
```
- 와 같이 설정할 수 있습니다. 이는 각 요청마다 지정된 프록시를 사용하게 합니다.

> output_loading_info -> bool, optional, defaults to False

- 불러오는 과정에서 누락된 키, 예상치 못한 키 및 오류 메세지를 포함하는 사전도 함께 반환할지 여부를 결정합니다. 디버깅이나 모델 상태를 검사할 때 유용합니다.

> local_files_only -> bool, optional, defaults to False

- **로컬 파일만을 참고하여 모델을 불러올지 여부를 결정**합니다. 즉, 모델을 인터넷에서 다운로드하는 것을 시도하지 않습니다. 네트워크 연결이 없거나 외부 다운로드를 원하지 않을 때 사용할 수 있습니다.

> revision -> str, optional, defaults to "main"

- 사용할 구체적인 모델 버전을 지정합니다. 이는 브랜치 이름, 태그 이름, 또는 커밋 ID가 될 수 있습니다. [[HuggingFace🤗]] 는 git 기반 시스템을 사용하여 모델과 다른 아티팩트를 저장하기 때문에, git에서 허용하는 모든 식별자를 사용할 수 있습니다.

> trust_remote_code -> bool, optional, defaults to False

- 사용자가 **허브에서 정의된 사용자 정의 모델 파일을 실행할 수 있게 할지 여부를 결정**합니다. 이 옵션은 신뢰할 수 있는 저장소의 코드를 읽고, 그 코드를 실행하는 것이 안전한 경우에만 `True` 로 설정해야 합니다. 이는 허브에 있는 코드를 로컬 머신에서 실행하기 때문에 보안 상의 문제가 될 수 있습니다.

> code_revision -> str, optional, defaults to "main"

- 코드가 모델과 다른 저장소에 있는 경우 사용할 코드의 구체적인 리비전을 지정합니다. 이 또한 브랜치 이름, 태그 이름, 또는 커밋 ID가 될 수 있으며, git 기반 시스템을 사용합니다.

