`torch.nn.GroupNorm` 은 [[Pytorch]] 의 normalization layers 중 하나로, 주어진 입력을 <font color="#ffff00">그룹별로 정규화</font>하는 데 사용됩니다. 이 정규화 방법은 <font color="#00b050">nn.BatchNorm</font> 과 유사하지만, 배치 대신 그룹을 기반으로 합니다.

여기서 <font color="#ffff00">그룹</font>은 <font color="#ffff00">입력 채널의 일부</font>를 의미하며, 각 그룹에 대해 평균과 분산을 계산하고 이를 사용하여 입력을 정규화합니다. 그룹의 크기는 사용자가 지정할 수 있으며, 일반적으로 채널의 개수로 나누어진 그룹을 만들어 사용됩니다.

```python
torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
```

> num_groups -> int
- 채널을 나눌 그룹의 수를 나타냅니다. 입력 채널을 이 수로 나누어 그룹별로 정규화를 수행합니다.

> num_channels -> int
- 입력으로 예상되는 채널의 수를 나타냅니다. 이 값은 <font color="#ffff00">입력 텐서의 두 번째 차원의 크기와 동일</font>해야 합니다.

> eps -> float
- 안정성을 위해 분모에 추가되는 값입니다. 분모가 0이 되는 것을 방지하기 위해 작은 값을 더합니다. 기본값은 1e-5

> affine -> bool ( befault : True)
- 이 파라미터는 학습 가능한 채널별 스케일과 shift parameter를 사용할지 여부를 결정합니다.
- True로 설정하면, 각 채널에 대한 weight 와 bias 파라미터가 모델 학습 중에 학습됩니다. 가중치는 1로 초기화되고, 편향은 0으로 초기화됩니다.
- False로 설정하면, weight 및 bias가 사용되지 않으며, 단지 입력을 정규화합니다.

```python
import torch
import torch.nn as nn

in_feature = torch.randn(3, 6)
print(in_feature)

m = nn.GroupNorm(3, 6)
m = nn.GroupNorm(6, 6)
m = nn.GroupNorm(1, 6)

output = m(in_feature)
print(output)

```

```
tensor([[-0.7823,  0.2508,  0.1282,  0.8421,  1.2992,  0.2108],
        [-1.4021, -0.9197, -0.7490,  0.3319, -1.1686, -0.1054],
        [ 0.1583,  1.1226,  0.0429,  1.0379,  0.8450, -0.7788]])
        
tensor([[-1.7152, -0.1147, -0.3046,  0.8014,  1.5095, -0.1766],
        [-1.2177, -0.4165, -0.1332,  1.6618, -0.8300,  0.9357],
        [-0.3671,  1.0697, -0.5390,  0.9436,  0.6561, -1.7633]],
```
