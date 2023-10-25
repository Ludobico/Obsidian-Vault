`torch.nn.Sequential` 은 [[Pytorch]] 에서 사용되는 <font color="#ffff00">신경망 계층을 순차적으로 연결하는 컨테이너</font>입니다. `Sequential` 은 다른 모듈을 간단하게 연결하여 신경망 모델을 구성하는 데 사용됩니다.

```python
# Using Sequential to create a small model. When `model` is run,
# input will first be passed to `Conv2d(1,20,5)`. The output of
# `Conv2d(1,20,5)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Conv2d(20,64,5)`. Finally, the output of
# `Conv2d(20,64,5)` will be used as input to the second `ReLU`
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

```python
import torch.nn as nn

# 간단한 순차적 모델 정의
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 모델에 데이터 전달
input_data = torch.randn(32, 128)
output = model(input_data)
```

`Sequential` 을 사용하면 신경망의 계층을 더 간편하게 구성할 수 있으며, 복잡한 모델을 간결하게 표현할 수 있습니다.

[[torch.nn.ModuleList]] 와의 주요 차이점은 `torch.nn.Sequential` 은 <font color="#ffff00">모듈을 순차적으로 연결</font>하는 반면 `ModuleList` 는 모듈을 순차적으로 연결하지 않고 List 형태로 저장한다는 것입니다.

