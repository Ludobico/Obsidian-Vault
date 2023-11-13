TrainingArguments 는 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 <font color="#ffff00">모델 훈련에 필요한 매개변수와 설정을 제어하는 클래스</font>입니다. 이 클래스를 사용하여 training loop 및 환경 설정을 구성할 수 있습니다. 아래는 주요 매개변수에 대한 설명입니다.

> output_dir -> str
- 모델의 결과와 체크포인트가 저장되는 경로를 설정합니다.

> learning_rate -> float, (optional), Default : 5e-5
- [[AdamW]] 옵티마이저에 사용되는 초기 학습률입니다.

> per_device_train_batch_size -> int, (optional), Default : 8
- 훈련 시 GPU 또는 CPU에 사용되는 배치사이즈입니다.

> per_device_eval_batch_size -> int, (optional), Default : 8
- 평가 시 GPU 또는 CPU에 사용되는 배치사이즈입니다.

> num_train_epochs -> float, (optional), Default : 3.0
- 총 학습 에포크 수입니다. int 형식이 아닐 시, 학습이 종료되기 전 마지막 에포크의 소숫점 이하 백분율로 표시됩니다.

> weight_decay -> float, (optional), Default : 0
- 0이 아닐 시, 바이어스와 [[AdamW]] 옵티마이저에 사용되는 LayerNorm 에 사용되는 가중치를 제외한 모든 레이어에 weight decay를 적용합니다.

> evaluation_strategy -> str, (optional), Default : "no"
- 훈련 중에 사용할 evaluation strategy 입니다. 사용가능한 값으로는
- "no" : 훈련 중 평가를 수행하지 않습니다.
- "strps" : eval_steps 마다 평가합니다.
- "epoch" : 1에포크마다 평가합니다.

