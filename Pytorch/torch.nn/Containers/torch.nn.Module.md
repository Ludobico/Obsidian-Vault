`torch.nn.Module` 은 [[Pytorch]] 에서 사용되는 신경망 모듈의 기본 클래스입니다. <font color="#ffff00">이 클래스를 상속하여 커스텀 모델을 생성하거나 파이토치의 기본 모델 클래스를 확장할 수 있습니다.</font> 모든 파이토치 모델은 `torch.nn.Module` 을 상속합니다.

`torch.nn.Module` 은 신경망 모듈의 기본 구조를 제공하며 다음과 같은 중요한 역할을 합니다.

1. <font color="#ffc000">모델 구성</font>
---
`torch.nn.Module` 을 상속한 클래스는 모델을 구성하는 계층(layer) 및 연산(operation)을 정의할 수 있습니다. 예를 들어, 합성곱 계층, 순환 신경망 계층, 완전연결 계층 등을 정의할 수 있습니다.

2. <font color="#ffc000">모듈 간의 중첩</font>
---
`torch.nn.Module` 은 다른 `torch.nn.Module` 객체를 포함할 수 있으며 이를 통해 모듈 간의 중첩이 가능합니다. 이러한 중첩 구조를 사용하여 복잡한 모델을 구성할 수 있습니다.

3. <font color="#ffc000">파라미터 관리</font>
---
`torch.nn.Module` 은 내부 파라미터를 관리하고 이러한 파라미터를 GPU로 이동하거나 저장할 수 있도록 도와줍니다. 이를 통해 모델의 학습 가능한 가중치 및 편향을 쉽게 관리할 수 있습니다.

4. <font color="#ffc000">순전파 및 역전파</font>
---
`torch.nn.Module` 을 상속한 클래스는 `forward` 메서드를 구현하여 [[Feed Forward propagation]] 연산을 정의합니다. 이러한 순전파는 모델에 입력 데이터를 주고 출력을 생성하는 과정을 의미하며, 이러한 순전파에 대한 [[Backward propagation]] 연산은 파이토치의 자동 미분 기능을 활용하여 자동으로 수행됩니다.

5. <font color="#ffc000">모듈 관리 및 중첩</font>
---
`torch.nn.Module` 을 상속한 클래스에서는 모듈 관리를 위해 `nn.Module` 을 중첩 및 조합하여 복잡한 모델을 구성할 수 있습니다. 이를 통해 모델의 구조를 계층적으로 정의하고 관리할 수 있습니다.

6. <font color="#ffc000">분산 학습 및 모델 저장</font>
---
`torch.nn.Module` 은 모델을 분산 학습에 사용하거나 모델의 상태를 저장하고 불러오는 데 도움을 줍니다.

예를 들어, 다음과 같이 `torch.nn.Module` 클래스를 상속하여 간단한 신경망 모델을 정의할 수 있습니다.
```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
  def __init__(self, inputs):
    super(MLP, self).__init__()
    # 계층 정의
    self.layer = nn.Linear(inputs, 1)
    # 활성화 함수 정의
    self.activation = nn.Sigmoid()

  def forward(self, x):
    x = self.layer(x)
    x = self.activation(x)
    return x
```

