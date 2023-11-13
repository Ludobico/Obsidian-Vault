DataCollatorWithPadding 은 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 제공되는 클래스 중 하나로, <font color="#ffff00">Train 데이터를 모델에 입력으로 전달하기 전에 패딩을 추가하는 역할</font>을 합니다. 이 클래스는 [[Pytorch]] 및 TensorFlow와 같은 다양한 딥러닝 프레임워크에서 사용할 수 있습니다.

일반적으로 Train 데이터의 시퀀스 길이가 다양하다면, 미니배치를 구성할 때 각 시퀀스의 길이를 동일하게 맞추는 것이 편리합니다. 이를 위해 패딩이 필요하며, DataCollatorWithPadding 는 이를 처리하는 데 도움을 줍니다.

```python
from transformers import DataCollatorWithPadding, BertTokenizer, BertForSequenceClassification

# 모델 및 토크나이저 초기화
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 훈련 데이터 예시
train_data = [
    {"text": "This is a positive example.", "label": 1},
    {"text": "Another example with a very long sequence.", "label": 0},
    # ... 다른 훈련 데이터들
]

# 토큰화 및 패딩 처리를 위한 DataCollatorWithPadding 초기화
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 데이터를 미니배치로 구성
batch = data_collator(train_data)

```

DataCollatorWithPadding 은 토큰화된 입력 데이터의 패딩을 자동으로 처리하여 각 미니배치 내에서 시퀀스의 길이를 동일하게 만듭니다. 이 미니배치는 모델에 직접 전달 할 수 있습니다.

이 예제에서는 사용된 모델과 [[Tokenizer]]는 BERT를 기반으로 한 예시입니다. 실제로 사용할 모델 및 토크나이저는 작업과 데이터에 따라 다를 수 있습니다.

주요 파라미터로는

> tokenizer -> PreTrainedTokenizer or PreTrainedTokenizerFast
- 텍스트 데이터를 모델이 이해할 수 있는 토큰으로 변환하는 역할을 하는 [[Tokenizer]] 객체입니다. 이는 [[transformers]] 라이브러리에서 제공하는 PreTrainedTokenizer 또는 PreTrainedTokenizerFast 인스턴스여야합니다. 위 인스턴스는 [[AutoTokenizer]] 로 제작할 수 있습니다.

> padding -> bool, str or PaddingStrategy, (optional), Default : True
- 시퀀스의 패딩을 처리하는 방식을 선택하는 매개변수입니다.
- <font color="#00b050">True</font> 또는 <font color="#00b050">longest</font> 일 경우 미니배치 내에서 가장 긴 시퀀스의 길이에 맞추어 패딩합니다.
- <font color="#00b050">max_length</font> 인자가 주어지면 지정된 최대 길이에 맞추어 패딩합니다.
- <font color="#00b050">False</font> 또는 <font color="#00b050">do_not_pad</font> 패딩하지 않고 서로 다른 길이의 시퀀스를 가진 미니배치를 생성합니다.

> max_length -> int, (optional)
- padding이 max_length로 설정된 경우 사용되며, 패딩된 시퀀스의 최대 길이를 지정합니다.

> pad_to_multiple_of -> int, (optional)
- 시퀀스를 주어진 값의 배수로 패딩합니다. 주로 NVIDIA의 Tensor Cores 를 활용하기 위해 사용됩니다.

> return_tensors -> str, (optional), Default : pt
- 반환되는 텐서의 유형을 지정합니다.
- np : [[Numpy]] 배열로 반환합니다.
- pt : [[Pytorch]] 텐서로 반환합니다.
- tf : TensorFlow 텐서로 반환합니다.

