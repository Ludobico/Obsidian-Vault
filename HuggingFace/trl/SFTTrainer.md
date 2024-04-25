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

