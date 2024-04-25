<font color="#ffff00">prepare_model_for_kbit_training</font> 메소드는 트랜스포머 모델을 효율적으로 훈련하기 위한 준비 과정을 제공하는 함수입니다. 이 메소드는 **모델의 메모리 사용량을 줄이는 대신, [[Backward propagation]] 시 속도가 느려지는 트레이드오프를 감수하면서 그래디언트 체크포인팅을 사용할 수 있는 옵션을 포함** 하고 있습니다.

> model -> [[transformers]].PreTrainedModel
- 라이브러리에서 로드된 사전 훈련된 모델입니다.

> use_gradient_checkpointing -> bool, optional, Default : True
- 이 옵션을 `True` 로 설정하면, 메모리 절약을 위해 그래디언트 체크포인팅을 사용합니다. 그 결과 역전파 과정이 느려질 수 있습니다.

> gradient_checkpointing_kwargs -> dict, optional, Default : None
- 그래디언트 체크포인팅 함수에 전달할 키워드 인수입니다. 이 인수들은 [[Pytorch]] 의 `torch.utils.checkpoint.checkpoint` 메소드의 문서를 참고하여 더 자세한 정보를 얻을 수 있습ㅣ다. 이 파라미터는 최신 버전의 [[transformers]] (> 4.34.1)에서만 사용 가능합니다.

