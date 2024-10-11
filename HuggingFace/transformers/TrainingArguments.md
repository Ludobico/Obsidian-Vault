TrainingArguments 는 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 <font color="#ffff00">모델 훈련에 필요한 매개변수와 설정을 제어하는 클래스</font>입니다. 이 클래스를 사용하여 training loop 및 환경 설정을 구성할 수 있습니다. 아래는 주요 매개변수에 대한 설명입니다.

> output_dir -> str
- 모델의 결과와 체크포인트가 저장되는 경로를 설정합니다.

> learning_rate -> float, (optional), Default : 5e-5
- [[AdamW]] 옵티마이저에 사용되는 초기 학습률입니다.

> logging_steps -> int or float, (optional), Default : 500
- 학습 과정 중에 로그를 출력하는 빈도를 결정하는 매개변수입니다.
- 이 매개변수는 학습 과정에서 모델의 성능 및 진행 상황을 모니터링하는 데 사용됩니다.
- 기본적으로 <font color="#ffff00">500</font> 으로 설정되어 있으며, 매 500번의 steps 마다 로그가 출력됩니다.

> per_device_train_batch_size -> int, (optional), Default : 8
- 훈련 시 GPU 또는 CPU에 사용되는 배치사이즈입니다.

> per_device_eval_batch_size -> int, (optional), Default : 8
- 평가 시 GPU 또는 CPU에 사용되는 배치사이즈입니다.

> num_train_epochs -> float, (optional), Default : 3.0
- 총 학습 에포크 수입니다. int 형식이 아닐 시, 학습이 종료되기 전 마지막 에포크의 소숫점 이하 백분율로 표시됩니다.

> max_steps -> int, optional, Default : -1
- 훈련을 진행할 최대 스탭 수를 정의합니다. 이 값이 양수로 설정되면, 지정된 스텝 수에 도달할 때지 데이터셋을 여러 번 반복하여 훈련합니다. `num_train_epochs` 설정을 덮어쓰기 때문에, 이 값이 설정되면 에포크 수 설정은 무시됩니다.

> lr_scheduler_type -> str or [[SchedulerType]] , optional, Default : "linear"
- 학습률 스케줄러의 유형을 결정합니다. 가능한 값은 [[SchedulerType]] 을 참조하십시오, "linear", "cosine", "constant" 등 여러 스케줄러 타입을 선택할 수 있습니다.

> lr_scheduler_kwargs -> 'dict', optional, Default : {}
- 학습률 스케줄러에 전달할 추가적인 인자들입니다. 각 스케줄러 타입에 따라 다른 인자들이 요구될 수 있습니다.

> warmup_ratio -> float, optional, Default : 0.0
- 전체 훈련 스텝 중 초기 학습률을 서서히 증가시키는데 사용되는 스텝의 비율을 설정합니다. 이는 학습 초기에 모델의 학습률을 점차 증가시키는 방법으로, 안정적인 학습이 가능하게 합니다.

> warmup_steps -> int, optional, Default : 0
- 학습 초기에 학습률을 서서히 증가시키는데 사용할 스텝 수를 명시합니다. `warmup_ratio` 의 를 덮어쓰므로, 이 값이 설정되면 `warmup_ratio` 는 무시됩니다.

> log_level -> str, optional, Default : passive
- 메인 프로세스에 사용할 로깅레벨을 설정합니다. 가능한 선택지는
```
'debug', 'info', 'warning', 'error', 'critical', 'passive'
```
입니다. 'passive'가 설정되면 [[transformers]] 라이브러리의 기본 로깅 레벨("warning")을 유지합니다.

> logging_dir -> str, optional
- TensorBoard 로그를 저장할 디렉토리 경로입니다. 설정되지 않은 경우
```
output_dir/runs/CURRENT_DATETIME_HOSTNAME
```
형태로 기본 경로가 설정됩니다.

> weight_decay -> float, (optional), Default : 0
- 0이 아닐 시, 바이어스와 [[AdamW]] 옵티마이저에 사용되는 LayerNorm 에 사용되는 가중치를 제외한 모든 레이어에 weight decay를 적용합니다.

> evaluation_strategy -> str, (optional), Default : "no"
- 훈련 중에 사용할 evaluation strategy 입니다. 사용가능한 값으로는
- "no" : 훈련 중 평가를 수행하지 않습니다.
- "strps" : eval_steps 마다 평가합니다.
- "epoch" : 1에포크마다 평가합니다.

> do_train -> bool, (optional), Default: False
- 학습 여부를 결정하는 파라미터입니다.
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

> save_strategy -> str or [[IntervalStrategy]], optional, Default : "steps"
- 훈련 중 **체크포인트를 언제 저장할지 결정하는 전략** 입니다. 가능한 값으로는
	-  "no" : 훈련 중에는 체크포인트를 저장하지 않습니다.
	- "epoch" : 매 에포크의 끝에서 체크포인트를 저장합니다.
	- "steps" : 지정된 스텝마다 체크포인트를 저장합니다.
	

> save_steps -> int or float, optional, Default : 500
- `save_strategy` 가 "steps" 일 경우, 몇 번의 업데이트 스텝 후에 체크포인트를 저장할지 결정하는 값입니다. 정수 혹은 0과 1 사이의 부동소수점으로 설정할 수 있습니다. 1미만의 값은 전체 훈련스텝의 비율로 해석됩니다. 소숫점으로 할 시 아래와 같은 전략에 따라 체크포인트가 저장됩니다.
```python
save_stpes = 0.1, 전체 학습 steps : 100
# 전체 학습 스텝이 100으로 가정할때 0.1*100=10 이므로 10 steps 마다 저장

save_steps = 0.25, 전체 학습 steps : 1000
# 전체 학습 스텝이 1000이므로 0.25*1000, 250 steps 마다 저장
```

> save_total_limit -> int, optional
- 저장할 체크포인트의 최대개수를 제한합니다. 이 값이 설정되면, **오래된 체크포인트부터 삭제하 제한된 수의 최신 체크포인트만 유지**합니다. `load_best_model_at_end` 가 활성화되면, 최적의 모델 체크포인트와 가장 최근의 체크포인트들을 함께 보존합니다.

> save_safetensors -> bool, (optional), Default : True
- 이 파라미터를 통해 모델의 상태를 저장하고 로드할 떄 기본적으로 제공되는 [[Pytorch]]  의 `torch.load` 및 `torch.save` 대신에 SafeTensors를 사용할지 여부를 설정할 수 있습니다.

> save_on_each_node -> bool, optional, Default : False
- 멀티노드 분산 훈련을 할 때, 각 노드에 모델과 체크포인트를 저장할지, 아니면 main 노드에만 저장할지를 결정합니다. 모든 노드가 같은 스토리지를 사용할 경우, 파일명 충돌을 피하기 위해 이 옵션을 활성화하지 않는 것이 좋습니다.

> save_only_model -> bool, optional, Default : False
- 체크포인트 저장 시, 모델만 저장할지 아니면 옵티마이저, 스케줄러 및 난수 생성 상태도 함께 저장할지 결정합니다. 이 옵션이 `True` 일 경우, 저장 공간을 절약할 수 있지만 훈련을 제개할 수 없게 됩니다. 이 경우 모델은 [[from_pretrained]] 를 사용하여 로딩할 수만 있습니다.
 
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
- <font color="#ffff00">adamw_torch</font> : [[Pytorch]] 의 AdamW 옵티마이저입니다.
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

> gradient_checkpointing -> bool, (optional), defaults to False
- **메모리를 절약하기 위해** gradient checkpointing을 사용할지 여부를 결정하는 파라미터입니다.
- 일반적으로 역전파 과정에서는 중간 계산 결과를 메모리에 저장하여 나중에 gradient를 계산하는데 사용됩니다. 그러나 gradient checkpointing을 사용하면 <font color="#ffff00">중간 계산 결과를 메모리에 저장하지 않고 필요할때마다 다시 계산하여 메모리 사용량을 줄일 수 있습니다</font>. 
- bench mark 결과를 보면 **연산 시간이 25% 가량 증가한 대신 메모리 사용량이 60% 가량 줄었다는 내용이 있습니다.**
![[Pasted image 20240205112100.png]]

> load_best_model_at_end -> bool, optional, Default : False
- train이 종료된 후 **가장 좋은 성능을 보인 모델을 자동으로 로드할지 결정하는 설정**입니다.
- 훈련 동안 모델의 성능이 검증 데이터셋에 대해 주기적으로 평가됩니다. 이 때 사용되는 매트릭은 일반적으로 설정에서 지정하거나 모델의 구성에 따라 자동으로 결정됩니다.
- `save_total_limit` 설정과 연동되어, 저장된 체크포인트의 수가 제한된 경우 최적의 체크포인트는 항상 보존됩니다. 예를 들어, `save_total_limit` 가 5로 설정되어있고 `load_best_model_at_end` 가 활성화되어 있다면, 최적의 모델과 가장 최근에 4개의 체크포인트가 보존됩니다. 이는 저장 공간을 효율적으로 사용하면서도, 최적의 모델을 손실하지 않기 위함입니다.

