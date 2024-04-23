<font color="#ffff00">from_pretrained() </font> 메소드는 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 사전 훈련된 모델을 불러오는 데 사용됩니다. 이 메소드는 다양한 매개변수를 통해 모델의 불러오기 및 설정을 맞춤화할 수 있습니다. 아래는 <font color="#ffff00">from_pretrained() </font> 메소드의 주요 매개변수들에 대한 설명입니다.

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

