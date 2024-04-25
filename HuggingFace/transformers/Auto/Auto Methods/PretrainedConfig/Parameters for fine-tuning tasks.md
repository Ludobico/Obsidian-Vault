> architentures -> List[str], (optional)
- 모델의 사전 학습 가중치와 함께 사용할 수 있는 모델 아키텍처의 목록입니다. 이 파라미터는 현재 모델에 적용 가능한 아키텍처를 지정하는데 사용합니다.
- input 값으로 리스트안 
```python
input_values = ["Hello", "World", "Python"]
```

> finetuning_task -> str, (optional)
- 모델을 파인튜닝하는데 사용되는 작업의 이름입니다. 이 파라미터는 원본 (TensorFlow 또는 [[Pytorch]]) 체크포인트로부터 변환할때 유용하며, <font color="#ffff00">파인튜닝 작업의 이름을 지정</font>하는데 사용됩니다.

> id2label -> Dict[int, str], (optional)
- 색인에서 레이블로 매핑하는 딕셔너리입니다. 이것은 모델의 출력에서 예측 된 클래스 레이블을 해석하는데 사용됩니다.

> label2id -> Dict[str, int], (optional)
- 레이블에서 색인으로 매핑하는 딕셔너리입니다. 이것은 레이블을 모델의 입력 또는 출력에서 인덱싱하는데 사용됩니다.

> num_labels -> int, (optional)
- 모델의 마지막 레이어에 추가할 레이블 수입니다. <font color="#ffff00">일반적으로 분류 작업의 경우 사용되며, 출력 클래스 수를 나타냅니다.</font>

> task_specific_params -> Dict[str, Any], (optional)
- 현재 작업에 대해 저장할 추가 키워드 인수입니다. 이는 특정 작업에 대한 추가 매개변수를 저장하고 전달할 때 유용합니다.

> problem_type -> str, (optional)
- `XxxForSequenceClassification` 모델에 대한 문제 유형을 지정합니다. `Xxx`는 모델의 이름에 따라 다릅니다. 문제 유형은
	- regression
	- single_label_classification
	- multi_label_classification
- 중 하나로 설정됩니다.


