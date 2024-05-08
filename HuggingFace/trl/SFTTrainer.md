- [[#difference between Trainer and SFTTrainer|difference between Trainer and SFTTrainer]]
- [[#Optimizer.pt|Optimizer.pt]]


[[HuggingFace🤗]] 의 <font color="#ffff00">SFTTrainer(Supervised Fine-Tuning Trainer)</font> 는 **대화형 언어 모델을 효과적으로 미세조정하기 위해 설계된 트레이너**입니다. 주요 특징은 다음과 같습니다.

1. Contrastive Language Modeling : SFTTrainer는 전통적인 교사 강요 방식 대신 누적 교정 전략을 사용합니다. 이 전략에서는 모델 출력과 실제 레이블 간의 유사도를 최대화하도록 학습합니다. 이를 통해 모델이 보다 일관된 응답을 생성할 수 있습니다.

2. 대화 데이터 전처리 : SFTTainer는 대화 형식의 데이터(질문-답변 쌍)를 효율적으로 전처리할 수 있습니다. 입력데이터는 "query : <query_text> response : <response_text>" 형식으로 변환하여 모델에 제공합니다.

3. 다양한 프롬프트 설정 : SFTTrainer는 프롬프트 설정을 통해 모델의 형식을 제어할 수 있습니다. 예를 들어 "question : <query_text> answer : " 와 같은 프롬프트를 사용하면 모델이 질문에 대한 답변 형식으로 출력하도록 할 수 있습니다.

4. 최대 길이 제한 : 대화 모델은 종종 긴 문맥을 처리해야 하므로, SFTTrainer는 `max_len` 파라미터를 통해 입력 시퀀스의 최대 길이를 제한할 수 있습니다. 이렇게 하면 모델이 문맥을 더 효과적으로 학습할 수 있습니다.

5. 생성 하이퍼파라미터 : SFTTrainer는 `max_length` , `min_length`, `repetition_penalty` 등의 하이퍼파라미터를 제공하여 모델의 텍스트 생성 품질을 세부적으로 제어할 수 있습니다.

6. 샘플링 전략 : 모델 출력의 다양성을 높이기 위해 SFTTrainer는 다양한 샘플링 전략 ([[top_k]], [[top_p]], 빔 서치 등)을 지원합니다.

7. 멀티 GPU/TPU 학습 : SFTTrainer는 데이터 병렬화를 통해 여러 GPU 또는 TPU에서 대규모 모델을 효율적으로 학습할 수 있습니다.

## difference between Trainer and SFTTrainer
---

[[HuggingFace🤗]] 의 [[transformers]] 에서 제공하는 [[Trainer]] 와 SFTTrainer 클래스는 모두 언어 모델을 미세조정(fine-tuning) 하는데 사용되지만, 주요 차이점은 다음과 같습니다.

1. 학습 목적
- Trainer : 일반적인 언어 모델 파인튜닝에 사용됩니다. 다양한 작업(텍스트 생성, 분류, 요약 등)에 활용될 수 있습니다.
- SFTTrainer : 특히 대화형 언어 모델 파인튜닝에 최적화되어 있습니다. 순차 텍스트 생성 작업에 특화되어 있습니다.

2. 학습 전략
- Trainer : 교사 강요(Teacher Forcing) 전략을 사용합니다. 이는 이전 단계의 실제 레이블(ground truth) 을 다음 단계의 입력으로 사용하는 방식입니다.

- SFTTrainer : 누적 교정(Contrastive Language modeling) 전략을 사용합니다. 이는 모델 출력과 실제 레이블 간의 유사도를 최대화하도록 학습합니다.

3. 데이터 전처리
- Trainer : 일반적인 텍스트 데이터를 사용할 수 있습니다.

- SFTTrainer : 대화 형식의 데이터(질문-답변 쌍)를 전처리하여 사용합니다.

4. 하이퍼파라미터
- Trainer : 일반적인 언어 모델 미세조정을 위한 하이퍼파라미터를 제공합니다.

- SFTTrainer : 대화형 모델 학습에 특화된 추가 하이퍼파라미터(`max_len`, `truncate_longer_inputs` 등을 제공합니다.)

## Optimizer.pt
---
![[Pasted image 20240425160355.png]]

SFTTrainer를 사용하여 LLM 모델을 학습할 때 <font color="#ffff00">optimizer.pt</font> 파일의 용량이 매우 커지는 현상은 SFTTrainer의 특성때문입니다.

일반적인 [[Trainer]] 와 달리 SFTTrainer는 누적 교정(Contrastive Language modeling) 전략을 사용합니다. 이 전략에서는 모델의 출력과 실제 레이블 간의 유사도를 최대화하도록 학습하는데, 이를 위해 모델의 모든 출력 토큰에 대한 손실을 계산해야 합니다.

손실 계산은 위해서는 모델의 모든 중간 활성화(intermediate activations) 를 메모리에 저장해야 하는데, 이 활성화 값들이 **optimizer.pt 파일에 누적되어 저장되기 때문에 파일 크기가 매우 커지게 됩니다.**

반면 일반 Trainer는 교사 강요(Teacher forcing) 전략을 사용하므로, 각 단계마다 실제 레이블만 사용해 손실을 계산하면 되기 때문에 중간 활성화 값을 저장할 필요가 없습니다. 따라서 optimizer.pt 파일의 크기가 SFTTrainer에 비해 작습니다.

이러한 SFTTrainer 의 특성으로 인해 대용량 LLM 모델을 학습할 때 optimizer.pt 파일의 크기가 수십 기가바이트에 이를 수 있습니다. 

## Parameters
---
SFTTrainer의 파라미터로는 다음과 같습니다.

> model -> Union\[transformers.PretrainedModel, nn.Module, str\]
- 트레이닝에 사용될 모델을 지정합니다. 이 모델은 [[transformers]].PretrainedModel, [[torch.nn.Module]] 이거나 모델의 이름을 나타내는 문자열중 하나입니다. 문자열을 사용할 경우, 해당 모델은 캐시에서 로드되거나 다운로드 됩니다. 또한 PeftConfig 객체가 `peft_config` 인자로 전달되면, 모델은 [[PeftModel]] 로 변환됩니다.

> args -> [[TrainingArguments]] , optional
- 학습 중에 조정할 수 있는 다양한 학습 매개변수를 설정하는데 사용됩니다. [[TrainingArguments]] 는 학습 시간, 배치 크기, 학습률 등을 포함한 다양한 설정을 제공합니다.

> data_collator -> [[DataCollator]] , optional 
- 데이터 콜레이터는 학습 데이터 배치를 자동으로 생성하는 데 사용됩니다. 예를 들어, 토크나이징 후 패딩 처리를 자동화하는 등의 작업을 처리할 수 있습니다.

> train_dataset -> [[datasets]], optional
- 학습에 사용될 데이터셋을 지정합니다. 이 외에도 [[trl]] 의 `ConstantLengthDataset` 을 사용하여 데터셋을 생성하는 것이 권장됩니다.

> eval_dataset -> optional, Union, [[datasets]] , Dict[str, datasets]
- 평가에 사용될 데이터셋입니다. 평가 시 여러 데이터셋을 사용할 경우 딕셔너리 형태로 제공할 수 있으며, 각 키에 해당하는 데이터셋이 평가에 사용됩니다. 여기서도 `train_dataset` 과 같이 [[trl]] 의 `constantLengthDataset` 을 권장합니다.

> tokenizer -> [[PreTrainedTokenizer]] , optional
- 텍스트 데이터를 모델이 처리할 수 있는 형태로 변환하는 데 사용되는 토크나이저입니다. 지정하않을 경우 **모델과 연관된 기본 토크나이저가 사용됩니다.**

> model_list -> Callable, PreTrainedModel
- 모델을 초기화하는 함수입니다. 이 함수는 아무런 인자 없이 호출되며, 새로운 `transformers.PreTrainedModel` 인스턴스를 반환해야 합니다. 이는 특히 교차 검증과 같이 여러번의 학습이 필요할 때 유용합니다.

> compute_metrics -> Callable, transformers.EvalPrediction, Dict, optional
- 평가 중에 메트릭을 계산하는 함수입니다. 이 함수는 `transformers.EvalPrediction` 을 입력으로 받고, 메트릭 이름과 그 값을 매핑한 딕셔너리를 반환해야 합니다.

> callbacks -> List, transformers.TrainerCallback
- 학습 과정에서 호출될 콜백 함수들의 리스트입니다.

> optimizers -> Tuple (torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR)
- 이 튜플은 학습에 사용될 옵티마이저와 학습률 스케줄러를 지정합니다. 옵티마이저는 모델의 파라미터를 업데이트하는 데 사용되며, 스케줄러는 학습 동안 학습률을 조정하는 데 사용됩니다.

> preprocess_logits_for_metrics -> Callable, [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] 
- 매트릭을 계산하기 전에 로짓(모델 출력의 원시 값)을 전처리하는 함수입니다. 이 함수는 로짓과 레이블을 입력으로 받아 처리된 로짓을 반환합니다.

> peft_config -> PeftConfig, optional
- [[PeftModel]] 을 초기화하는 데 사용되는 `PeftConfig` 객체입니다. 이 설정을 통해 모델의 특정 성능 특성을 조정할 수 있습니다.

> dataset_text_filed -> str, optional
- 데이터셋의 텍스트 필드 이름입니다. 이 필드는 `ConstantLengthDataset` 을 **자동으로 생성할 때 사용**됩니다.

> formatting_func -> Callable, optional
- `ConstantLengthDataset` 생성 시 사용할 포맷팅 함수입니다. 이 함수는 **데이터셋의 항목을 적절한 형태로 변환**하는 데 사용됩니다.

> max_seq_length -> int, optional, Default : 512
- `ConstantLengthDataset` 및 데이터셋 생성에 사용될 최대 시퀀스 길이입니다. 기본값은 512입니다.

> infinite -> bool, optional
- 데이터셋을 무한 반복할지 여부를 지정합니다. 기본값은 `False` 입니다.

> num_of_sequences -> int, optional
- `ConstantLengthDataset` 에 사용될 스퀀스의 수입니다. 기본값은 `1024` 입니다.

> chars_per_token -> float, optional
- 토큰 당 사용할 문자 수입니다. 이 값은 데이터셋의 효율적인 구성을 돕습니다. 기본값은 `3.6` 입ㅣ다.

> packing -> bool, optional
- 시퀀스를 패킹할지 여부를 지정합니다. 이는 주로 `dataset_text_filed` 가 지정되었을 때 사용합ㅣ다.

> dataset_num_proc -> int, optional
- 데이터 토크나이징에 사용될 [[worker]]의 수입니다. `packing=False` 인 경우에만 사용됩니다. 기본값은 `None` 입니다.

> dataset_batch_size -> int
- 한 번에 토크나이즈할 배치 사이즈입니다. 이 값이 0이하거나 `None` 일 경우, 전체 데이터셋을 단일 배치로 처리합니다. 기본값은 1000입니다.

> neftune_noise_alpha -> float, optional
- NEFTune 노이즈 임베딩을 활성화할지 여부입니다. 이 설정은 SFTtraining의 지시적 미세조정에있어 모델 성능을 크게 향상시킬 수 있습니다.

> model_init_kwargs -> dict, optional
- 모델 초기화의 추가적인 맥변수를 제공합니다.

> dataset_kwargs -> dict, optional
- 데이터셋 생성에 추가적인 매개변수를 제공합니다.

> eval_packing -> bool, optional
- 평가 데이터셋 패킹에 추가적인 매개변수를 제공합니다.

