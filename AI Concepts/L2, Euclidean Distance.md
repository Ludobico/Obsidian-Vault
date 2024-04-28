![[Untitled (1).png]]

L2는 두 벡터 또는 데이터 포인트 간의 <font color="#ffff00">제곱 차이를 제곱근으로 계산하는 방법</font>입니다. L2 손실은 제곱차이의 합으로 계산되며, 이는 이상치에 민감하고 작은 가중치 값을 가지는 특징을 가지고 있습니다. L2 손실은 작은 차이에 대해 더 작은 패널티를 부과하는 경향이 있습니다.

```python
import torch

def l2_distance(vector1, vector2)-> torch.Tensor:
  return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2)))

vector1 = torch.tensor([1,2,3])
vector2 = torch.tensor([4,5,6])

L2_dist = l2_distance(vector1, vector2)

print("L2 Distance : {0}".format(L2_dist.item()))
```

```
L2 Distance : 5.196152210235596
```

계산식으로는 다음과 같이 진행됩니다.

먼저 [[Pytorch/torch.Tensor/torch.Tensor]] 로 만들어진 각 텐서의 절댓값 차이를 계산합니다.
$$\left\vert [1,2,3] - [4,5,6] \right\vert$$

각 차이로 계산된 텐서를 제곱합니다.
$$\left\vert [3^2, 3^2, 3^2] \right\vert$$

각 텐서를 더합니다.
$$27$$

주어진 텐서(스칼라)에 대해 제곱근을 취합니다.
$$\sqrt 27 \therefore 5.19615221...$$

