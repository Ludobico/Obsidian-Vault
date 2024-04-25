- [[#Methods|Methods]]
	- [[#Methods#train()|train()]]


[[HuggingFace🤗]] 의 [[transformers]] 라이브러리의 Trainer 클래스는 모<font color="#ffff00">델 훈련을 단순화하고 추상화하는 데 사용되는 클래스</font>입니다. Trainer는 [[Pytorch]] 또는 TensorFlow를 기반으로 하는 모델의 훈련과 평가를 관리하며, 훈련 중에 발생하는 다양한 작업을 처리합니다.

Trainer 클래스의 인스턴스를 만들 때는 훈련에 필요한 여러 파라미터를 지정해야 합니다. 주요 파라미터 중 일부는 다음과 같습니다.

> model -> PreTrainedModel or torch.nn.Module, (optional)
- 훈련할 모델의 인스턴스입니다.

> args -> TrainingArguments, (optional)
- 훈련에 필요한 여러 파라미터 및 설정을 포함하는 [[TrainingArguments]] 클래스의 인스턴스입니다. 이 클래스에는 <font color="#ffff00">epoch, 배치 크기, 로그 출력 디렉토리</font> 등을 설정할 수 있는 다양한 파라미터가 있습니다. 

> data_collator -> DataCollator, (optional)
- <font color="#ffff00">미</font><font color="#ffff00">니배치의 데이터를 처리</font>하기 위한 [[DataCollator]] 또는 [[DataCollatorWithPadding]] 의 인스턴스입니다. train 데이터를 모델 입력 형식을 변환하고 패딩을 추가하는 데 사용됩니다.

> tokenizer -> PreTrainedTokenizerBase, (optional)
- 모델의 토크나이저입니다. 주어진 텍스트 데이터를 모델이 이해할 수 있는 형식으로 토큰화하는 데 사용됩니다.

> compute_metrics -> Callable[[EvalPrediction], Dict], (optional)
- 평가 매트릭스를 계산하는데 사용되는 함수입니다. [[EvalPrediction]] 을 파라미터로 받으며, 매트릭 값을 문자열 딕셔너리로 반환합니다.

> callbacks -> List of TrainerCallback, (optional)
- 사용자 지정 training loop를 설정하기 위한 콜백 리스트입니다. 

> train_dataset -> torch.utils.data.Dataset of torch.utils.data.IterableDataset, (optional)
- 학습에 사용되는 데이터셋입니다. 데이터셋에서 컬럼의 이름은 사용되지 않으며, [[Feed Forward propagation]] 메서드는 자동으로 제거됩니다.

> eval_dataset -> Union[torch.utils.data.Dataset, Dict[torch.utils.data.Dataset]], (optional)
- 평가에 사용되는 데이터셋입니다. 데이터셋에서 컬럼의 이름은 사용되지 않으며, [[Feed Forward propagation]] 메서드는 자동으로 제거됩니다. 만약 파라미터가 딕셔너리로 주어진다면, 각각의 딕셔너리 키 값 앞에 이름이 붙여집니다.

## Methods
---

### train()

> resume_from_checkpoint -> str or bool, optional
- 이 파라미터는 이전에 저장된 체크포인트에서 훈련을 제개할지 여부를 결정합니다.
- 문자열(str) 이 제공된다면 그 경로에 있는 체크포인트 파일에서 모델,옵티마이저,스케줄러의 상태를 불러와 훈련을 재게합니다.
- 불리언(bool) 값이 `True` 로 설정된 경우, [[TrainingArguments]] 의 `output_dir` 에 저장된 마지막 체크포인트를 불러옵니다.
- 체크포인트를 통해 훈련을 제개하는 기능은 훈련 중에 예기치 못한 중단이 발생했을 때 유용하게 사용할 수 있습니다.

> trial -> optuna, Trial or Dict[str, Any], optional
- `optuna.Trial` 객체가 주어지면, 해당 트라이얼을 사용하여 하이퍼파라미터를 최적화합니다.
- 딕셔너리 형태로 주어지면, 해당 딕셔너리에 명시된 하이퍼파라미터 값들을 사용하여 훈련을 진행합니다.
- 이를 통해 다양한 하이퍼파라미터 설정을 실험하여 최적의 모델 구성을 찾을 수 있습니다.

> ignore_keys_for_eval -> List[str], optional
- 이는 모델이 다양한 출력을 가질 때, 특정 출력들이 평가 과정에 포함되지 않도록 설정할 수 있습니다.
- 예를 들어, 일부 출력은 학습에만 유용하고 평가 시에는 필요 없는 정보일 수 있습니다.

> kwrags -> Dict[str, Any], optional
- 추가적인 키워드 인자들을 사용하여, 사용되지 않는 또는 더 이상 권장되지 않는 인자들을 숨길 수 있습니다. 이는 함수의 시그니처를 깔끔하게 유지하면서도 필요에 따라 유연하게 파라미터를 조정할 수 있게 해줍니다.

