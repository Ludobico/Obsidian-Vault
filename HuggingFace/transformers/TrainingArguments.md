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

> do_train -> bool, (optional), Default: False
- 학습 여부를 결정ㅎ는 파라미터입니다.
- 기본값은 `False` 이며, 학습을 수행하지 않습니다.
- 이 파라미터는 [[Trainer]] 에 직접적으로 사용되지 않고, 학습/평가 스크립트에서 사용됩니다.

> do_eval -> bool, (optional)
- 검증 세트에 대한 평가 여부를 결정하는 파라미터입니다.
- `evaluation_strategy` 가 `no` 가 아닌 경우, 기본적으로 `True`로 설정됩니다.
- 이 파라미터는 [[Trainer]] 에 직접적으로 사용되지 않고, 학습/평가 스크립트에서 사용됩니다.

> do_predict -> bool, (optional), Default: False
- 테스트 세트에 대한 예측 여부를 결정하는 파라미터입니다.
- 기본값은 `False` 이며, 예측을 수행하지 않습니다.
- 이 파라미터는 [[Trainer]] 에 직접적으로 사용되지 않고, 학습/평가 스크립트에서 사용됩니다.

> save_safetensors -> bool, (optional), Default : True
- 이 파라미터를 통해 모델의 상태를 저장하고 로드할 떄 기본적으로 제공되는 `torch.load` 및 `torch.save` 대신에 SafeTensors를 사용할지 여부를 설정할 수 있습니다.

> use_cpu -> bool, (optional), Default : False
- CPU 사용 여부를 결정하는 파라미터입니다.
- 기본값은 `False` 이며, CUDA 또는 MPS 장치가 사용가능한 경우에는 해당 장치를 사용합니다.
- 만약 `use_cpu` 를 `True` 로 설정하면 CPU를 사용하여 모델을 학습하게 됩니다.
- 주로 CUDA를 사용할 수 없는 환경이거나, 병렬처리를 지원하지 않는 환경에서 모델을 실행할때 사용됩니다.

> seed -> int, (optional), Default : 42
- 학습을 시작할 때 설정할 난수 시드입니다.
- 기본값은 42입니다.
- 동일한 시드를 사용하면 학습 과정에서 재현 가능한 결과를 얻을 수 있습니다.

> data_seed -> int, (optional)
- 데이터 샘플러(data sampler)에 사용할 난수 시드입니다.
- 이 파라미터가 설정되지 않으면, 데이터 샘플링에 대한 난수는 `seed` 와 동일한 시드를 사용합니다.
- 모델의 시드와 독립적으로 데이터 샘플링의 재현성을 보장하기 위해 사용됩니다.

> optim -> str, (optional), Default : "adamw_torch"
- 옵티마이저를 지정하는데 사용되는 매개변수입니다. 종류는 아래와 같습니다.
- <font color="#ffff00">adamw_torch</font> : [[HuggingFace🤗]] 의 AdamW 옵티마이저입니다.
- <font color="#ffff00"> adamw_torch</font> : [[Pytorch]] 의 AdamW 옵티마이저입니다.
- <font color="#ffff00">adamw_torch_fused</font> : [[Pytorch]] 의 Fused AdamW 옵티마이저입니다.
- <font color="#ffff00">adamw_apex_fused</font> : NVIDIA Apex의 Fused AdamW 옵티마이저입니다.
- <font color="#ffff00">adamw_anyprecision</font> : 다양한 정밀도를 지원하는 AdamW 옵티마이저입니다.
- <font color="#ffff00">adafactor</font> : AdaFactor 옵티마이저입니다.

> gradient_accumulation_steps -> int, (optional), defaults to 1
- 한 번의 역전파 및 가중치 업데이트를 수행하기 전에 gradient를 누적하는데 사용됩니다.
- 예를 들어, gradient_accumulation_steps가 2로 설정되어 있다면, 두 번의 순전파 및 역전파가 이루어진 후에 gradient가 업데이트됩니다. 이렇게 하면 **메모리 사용량이 줄어들고, 더 큰 배치 크기를 사용할 수 있으며, 훈련 속도를 조절할 수 있습니다.**
- 예를 들어 `batch_size = 8` 대신 `batch_size = 2` 를 사용하고 `gradient_accumulation_steps = 4` 로 설정하면 아래와 같은 steps를 따르게 됩니다.
```
첫 번째 배치를 입력으로 사용하여 순전파를 수행합니다.
그라디언트를 계산하고 이를 누적합니다.
두 번째 배치를 입력으로 사용하여 순전파를 수행합니다.
그라디언트를 계산하고 이를 누적합니다.
세 번째 배치를 입력으로 사용하여 순전파를 수행합니다.
그라디언트를 계산하고 이를 누적합니다.
네 번째 배치를 입력으로 사용하여 순전파를 수행합니다.
그라디언트를 계산하고 이를 누적합니다.
총 4번의 배치를 처리한 후, 역전파 및 가중치 업데이트를 수행합니다.
```

