
- [[#transformers.get_scheduler|transformers.get_scheduler]]
- [[#get_constant_schedule|get_constant_schedule]]
- [[#get_constant_schedule_with_warmup|get_constant_schedule_with_warmup]]
- [[#get_cosine_schedule_with_warmup|get_cosine_schedule_with_warmup]]
- [[#get_cosine_with_hard_restarts_schedule_with_warmup|get_cosine_with_hard_restarts_schedule_with_warmup]]
- [[#get_linear_schedule_with_warmup|get_linear_schedule_with_warmup]]

## transformers.get_scheduler
---

> ( name: Unionoptimizer: Optimizernum_warmup_steps: Optional = Nonenum_training_steps: Optional = Nonescheduler_specific_kwargs: Optional = None )

`transformers.get_scheduler` 는 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 제공하는 함수입니다. 이 함수는 **다양한 학습율 스케줄러(learning rate scheduler)를 이름으로 가져올 수 있게 해줍니다.**

학습율 스케줄러는 모델 학습 과정에서 학습율을 조정하는 역할을 합니다. 적절한 학습율 스케줄링은 모델의 수렴 속도와 성능에 큰 영향을 미칩니다.

`get_scheduler` 함수는 다음과 같은 파라미터를 가집니다.

> name -> str or [[SchedulerType]] 
- 사용할 스케줄러의 이름 또는 타입을 문자열로 지정합니다.

> optimizer -> torch.optim.optimizer
- 학습에 사용될 [[Pytorch]] 옵티마이저 객체를 전달합니다.

> num_warmup_steps -> int, optional
- 웜업 단계의 스텝 수를 지정합니다. (일부 스케줄러에 필요)

> num_training_steps -> int, optional
- 전체 학습 단계의 스텝 수를 정수로 지정할 수 있습니다. (일부 스케줄러에 필요)

> scheduler_specific_kwargs -> dict, optional
- 특정 스케줄러에 필요한 추가 매개변수를 딕셔너리로 전달할 수 있습니다.

이 함수를 통해 사용자는 스케줄러의 이름만으로 원하는 스케줄러 객체를 쉽게 가져올 수 있으며, 학습 과정에 필요한 매개변수들도 지정할 수 있습니다. 이렇게 통일된 API를 제공함으로써 다양한 스케줄러를 일관되게 사용할 수 있습니다.

## get_constant_schedule
---

> ( optimizer: Optimizerlast_epoch: int = -1 )

`get_constant_schedule` 은 [[Pytorch]] 의 `torch.optim.lr_scheduler` 모듈에서 사용할 수 있는 학습률 스케줄러입닏. 이 스케줄러는 **학습 중 학습률을 일정하게 유지하는 방법을 제공**합니다. 사용자가 지정한 옵티마이저에 설정된 초기 학습률을 변경하지 않고 계속 유지하게 됩니다.

> optimizer -> torch.optim.Optimizer
- 학습률을 스케줄할 옵티마이저 객체입니다. 이 옵티마이저에 설정된 초기 학습률을 기준으로 학습률이 유지됩니다.

> last_epoch -> int, optional, default : -1
- 학습을 제개할 때 마지막으로 완료된 에포크의 인덱스를 지정합니다. 기본값은 `-1` 은 스케줄러 처음부터 시작됨을 의미합니다.

## get_constant_schedule_with_warmup
---

`get_constant_schedule_with_warmup` 은 학습률 스케줄러 중 하나로, **일정한 웜업 기간을 거친 후 학습률을 일정하게 유지하는 방법을 제공**합니다. 이 스케줄러는 특히 모델 학습 초기에 학습률을 점진적으로 증가시킴으로써 학습 과정을 안정화시키고자 할 때 유용합니다.

> optimizer -> torch.optim.Optimizer
- 학습률을 조정할 옵티마이저 객체입니다. 옵티마이저에 설정된 초기 학습률을 기준으로, 웜업 기간 동안 0에서 시작하여 이 학습률까지 선형적으로 증가합니다.

> num_warmup_steps -> int
- 웜업 기간 동안의 스텝(또는 에포크) 수를 지정합니다. 이 기간 동안 학습률은 0에서 시작하여 선형적으로 옵티마이저에 설정된 초기 학습률까지 증가합니다.

> last_epoch -> int, optional, default : -1
- 학습을 제개할 때 마지막으로 완료된 에포크의 인덱스를 지정합니다. 기본값은 `-1` 은 스케줄러 처음부터 시작됨을 의미합니다.

![[Pasted image 20240502155136.png]]

## get_cosine_schedule_with_warmup
---

`get_cosine_schedule_with_warmup` 은 **웜업 기간 동안 학습률을 점진적으로 증가시킨 후, 코사인 함수의 형태에 따라 학습률을 점차 감소**시키는 방법을 제공합니다. 이 스케줄러는 학습 초기에 모델을 안정적으로 훈련시키고자 하는 목적과 함께, 장기간 학습에서 발생할 수 있는 과적합을 피하면서 모델의 일반화 성능을 향상시키기 위해 설계되었습니다.

> optimizer -> torch.optim.Optimizer
- 학습률을 조정할 옵티마이저 객체입니다. 이 옵티마이저에 설정된 초기 학습률에서 시작하여 학습률 조정이 이루어집니다.

> num_warmup_steps -> int
- 웜업 기간 동안의 스텝(또는 에포크) 수를 지정합니다. 이 기간 동안 학습률은 0에서 시작하여 선형적으로 옵티마이저에 설정된 초기 학습률까지 증가합니다.

> num_training_steps -> int
- 총 훈련 스텝 수입니다. 이는 스케줄러가 언제 학습률 조정을 멈출지를 결정합니다.

> num_cycles -> float, optional, default : 0.5
- 코사인 스케줄의 주기 수를 나타냅니다. 기본값 `0.5` 는 한 주기의 절반인 반코사인 형태로 학습률이 최대치에서 0까지 감소합니다.

> last_epoch -> int, optional, default : -1
- 학습을 제개할 때 마지막으로 완료된 에포크의 인덱스를 지정합니다. 기본값은 `-1` 은 스케줄러 처음부터 시작됨을 의미합니다.

![[Pasted image 20240502155734.png]]

## get_cosine_with_hard_restarts_schedule_with_warmup
---

`get_cosine_with_hard_restarts_schedule_with_warmup` 은 학습률 스케줄러 중 하나로, **웜업 기간 후 콧인 함수에 기반한 학습률 감소와 함께 여러번의 하드 리스타트(hard restarts)를 포함**합니다. 이 스케줄러는 모델 학습 초기에 안정적인 웜업을 제공하고, 그 후 여러 사이클에 걸쳐 학습률이 감소하다가 특정 지점에서 갑자기 다시 최대치로 상승하는 특징을 가집니다. 이러한 방식은 지역 최소값(local minima)에서 벗어나 전역 최소값(global minima)을 찾는데 도움을 줄 수 있습니다.

> optimizer -> torch.optim.Optimizer
- 학습률을 조정할 옵티마이저 객체입니다. 이 옵티마이저에 설정된 초기 학습률에서 시작하여 학습률 조정이 이루어집니다.

> num_warmup_steps -> int
- 웜업 기간 동안의 스텝(또는 에포크) 수를 지정합니다. 이 기간 동안 학습률은 0에서 시작하여 선형적으로 옵티마이저에 설정된 초기 학습률까지 증가합니다.

> num_training_steps -> int
- 총 훈련 스텝 수입니다. 스케줄러가 학습률을 조정하는 전체 과정을 나타냅니다.

> num_cycles -> int, optional, defaut : 1
- 하드 리스타트의 횟수로, 학습 과정중 학습률이 여러 번 리셋되어 최대치로 다시 상승합니다. 각 사이클은 학습률이 점차 감소하다가 끝에서 다시 초기 값으로 상승합니다.

> last_epoch -> int, optional, default : -1
- 학습을 제개할 때 마지막으로 완료된 에포크의 인덱스를 지정합니다. 기본값은 `-1` 은 스케줄러 처음부터 시작됨을 의미합니다.

![[Pasted image 20240502160156.png]]

## get_linear_schedule_with_warmup
---

`get_linear_schedule_with_warmup`은 학습률 스케줄러로서 **특정 웜업기간 동안 학습률을 점진적으로 증가시킨후, 그 이후에는 학습률을 선형적으로 감소**시키는 방식을 제공합니다. 이 스케줄러는 학습 초기에는 학습률을 천천히 높이면서 모델을 안정적으로 학습시킬 수 있도록 돕고, 학습이 진행됨에 따라 학습률을 감소시켜 최적화 과정을 더욱 효과적으로 마무리하도록 설계되었습니다.

> optimizer -> torch.optim.Optimizer
- 학습률을 조정할 옵티마이저 객체입니다. 이 옵티마이저에 설정된 초기 학습률에서 시작하여 학습률 조정이 이루어집니다.

> num_warmup_steps -> int
- 웜업 기간 동안의 스텝(또는 에포크) 수를 지정합니다. 이 기간 동안 학습률은 0에서 시작하여 선형적으로 옵티마이저에 설정된 초기 학습률까지 증가합니다.

> num_training_steps -> int
- 총 훈련 스텝 수입니다. 스케줄러가 학습률을 조정하는 전체 과정을 나타냅니다.

> last_epoch -> int, optional, default : -1
- 학습을 제개할 때 마지막으로 완료된 에포크의 인덱스를 지정합니다. 기본값은 `-1` 은 스케줄러 처음부터 시작됨을 의미합니다.

![[Pasted image 20240502160411.png]]

