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

> 