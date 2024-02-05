![[Untitled 1.png]]

L1은 <font color="#ffff00">두 벡터 또는 데이터 포인트 간의 절대 차이를 계산하는 방법</font>입니다. L1 손실함수는 예측 값과 실제 값 사이의 차이를 계산할 때 사용됩니다. L1 손실은 **절대값의 합으로 계산** 되며, 이는 이상치에 덜 민감하고 더 많은 제로 값을 가지는 특징이 있습니다. L1 손실은 더 많은 가중치를 가진 이상치에 대해 더 큰 패널티를 부과하는 경향이 있습니다.

```python
import torch

def l1_distance(vector1, vector2):
  return torch.sum(torch.abs(vector1 - vector2))

vector1 = torch.tensor([1,2,3])
vector2 = torch.tensor([4,5,6])

L1_dist = l1_distance(vector1, vector2)

print("L1 Distance : {0}".format(L1_dist.item()))
```

```
L1 Distance : 9
```

