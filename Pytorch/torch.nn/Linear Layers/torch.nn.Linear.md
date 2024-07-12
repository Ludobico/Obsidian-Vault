- [[#Principle|Principle]]
	- [[#Principle#3D tensor input|3D tensor input]]
	- [[#Principle#4D tensor input|4D tensor input]]


`torch.nn.Linear` 모듈은 인공 신경망에서 사용되는 <font color="#ffff00">선형 변환(Linear Transformation)을 수행하는 모듈</font>입니다. 이 모듈은 입력 데이터를 선형 방정식을 통해 변환하고 편향(bias)를 더하는 역할을 합니다.

$$y = xW^T + b$$

> in_features -> int
- 입력 데이터의 특성 수 또는 차원 수를 나타내는 정수입니다.

> out_features -> int
- 출력 데이터의 특성 수 또는 차원 수를 나타내는 정수입니다.

> bias -> bool
- 편향을 사용할지 여부를 나타내는 불리언 값입니다. <font color="#ffc000">True</font> 로 설정하면 편향이 사용되고 <font color="#ffc000">False</font>로 설정하면 편향이 사용되지 않습니다.

```python
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
```

```bash
torch.Size([128, 30])
```


## Principle
---

다음은 `nn.linear(20,30)` 이고 입력텐서가 2차원 (n, 20) n=batch_size 일 경우 동작 원리입니다.

```python
import torch
import torch.nn as nn

# nn.Linear 모듈 생성
linear_layer = nn.Linear(20, 30)

# 임의의 입력 텐서 생성 (배치 크기 N은 5로 가정)
input_tensor = torch.randn(5, 20)

# 선형 변환 적용
output_tensor = linear_layer(input_tensor)

print("Input tensor shape:", input_tensor.shape)   # (5, 20)
print("Output tensor shape:", output_tensor.shape) # (5, 30)
```

`nn.linear` 는 다음과 같은 선형 변환을 수행합니다.

$$
y = xW^T + b
$$

- $x$ 는 입력 텐서이며, 크기는 <font color="#ffff00">(N, 20)</font> 입니다.
- $W$ 는 가중치 행렬이며, 크기는 <font color="#ffff00">(30,20)</font> 입니다.
- $b$ 는 편향 벡터이며, 크기는 <font color="#ffff00">(30)</font> 입니다.
- $y$ 는 출력 텐서이며, 크기는 <font color="#ffff00">(N, 30)</font> 입니다.

1. 크기 (N, 20) 인 입력 텐서가 주어집니다.
2. 크기 (30,20) 인 가중치 행렬 $W$ 는 텐서 $x$ 와 **곱해지기 전에 전치**되어 $W^T$ 가 됩니다. 전치된 가중치 행렬 $W^T$의 크기는 (20, 30)이 됩니다.
3. $x$ 와 $W^T$ 의 행렬 곱셈이 이루어져 크기 (N, 30)의 행렬이 생성됩니다.
4. 크기 (30) 인 편향 벡터 $b$ 는 브로드캐스팅되어 행렬 곱셈의 결과에 더해집니다.

### 3D tensor input

```python
import torch
import torch.nn as nn

# nn.Linear 모듈 생성
linear_layer = nn.Linear(20, 30)

# 임의의 입력 텐서 생성 (예: 배치 크기 5, 시퀀스 길이 10, 입력 차원 20)
input_tensor = torch.randn(5, 10, 20)

# 선형 변환 적용
output_tensor = linear_layer(input_tensor)

print("Input tensor shape:", input_tensor.shape)   # (5, 10, 20)
print("Output tensor shape:", output_tensor.shape) # (5, 10, 30)
```

### 4D tensor input

```python
import torch
import torch.nn as nn

# nn.Linear 모듈 생성
linear_layer = nn.Linear(20, 30)

# 임의의 입력 텐서 생성 (예: 배치 크기 5, 높이 8, 너비 10, 입력 차원 20)
input_tensor = torch.randn(5, 8, 10, 20)

# 선형 변환 적용
output_tensor = linear_layer(input_tensor)

print("Input tensor shape:", input_tensor.shape)   # (5, 8, 10, 20)
print("Output tensor shape:", output_tensor.shape) # (5, 8, 10, 30)
```

- 3차원 텐서 : 크기가 (N, S, 20)인 입력 텐서를 **(N * S, 20)으로 변형**한 후 선형 변환을 적용하여 (N * S , 30)의 출력을 얻고, 다시 (N, S, 30)으로 변형합니다.
- 4차원 텐서 : 크기가 (N, H, W, 20)인 입력 텐서를 **(N * H * W, 20)으로 변형**한 후 선형 변환을 적용하여 (N * H * W, 30)의 출력을 얻고, 다시 (N, H, W, 30)으로 변형합니다.

이처럼 `nn.linear` 모듈은 입력 텐서가 몇 차원이든 **마지막 두 차원에 대해 선형 변환을 적용**합니다. 이를 통해 3차원이나 4차원 텐서에 대해서도 일관된 방식으로 선형 변환을 수행할 수 있습니다.

