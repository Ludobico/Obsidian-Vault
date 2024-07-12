
> torch.nn.LayerNorm(_normalized_shape_, _eps=1e-05_, _elementwise_affine=True_, _bias=True_, _device=None_, _dtype=None_)

`torch.nn.LayerNorm` 은 입력 텐서의 미니 배치에 대해 **레이어 정규화를 적용**하는 [[Pytorch]] 모듈입니다. 이는 주어진 텐서의 마지막 `D` 차원에 걸쳐 평균과 표준편차를 계산하여, 각 요소를 정규화하고, optional로 학습 가능한 가중치와 편향을 추가합니다. 

## Basic

- <font color="#ffff00">레이어 정규화</font>는 신경망의 각 레이어에서 입력 값을 정규화하여 학습 속도를 높이고 모델의 성능을 향상시키기 위한 기법입니다.
- 이는 주어진 텐서의 마지막 `D` 차원에 대해 평균과 표준편차를 계산합니다.

$$
y = \frac{x - E[x]}{\sqrt{\text{Var}[x] + \epsilon}} \cdot \gamma + \beta
$$

- $E[x]$ 평균
- $\text{var}[x]$ 분산
- $\epsilon$ 수치적 안정성을 위한 작은 값
- $\gamma$ 가중치
- $\beta$ 편향

## Parameters

> normalized_shape
- 정규화할 텐서의 마지막 차원의 크기, 이는 정규화가 적용될 축을 정의합니다.
- 단일 정수값을 사용할 수도 있으며, 이는 마지막 차원의 크기로 취급됩니다.

> eps
- 분모에 더해지는 작은 값으로 수치적 안정성을 제공합니다. 기본값은 `1e-5` 입니다.

> elementwise_affine
- `True` 로 설정되면, 학습 가능한 가중치($\gamma$) 와 편향($\beta$) 를 사용합니다. 기본값은 `True` 입니다.

> bias
- `False` 로 설정하면, 가중치만 학습하고 편향은 학습하지 않습니다. 기본값은 `True` 입니다.

## Variable
- weight : 학습 가능한 가중치 $\gamma$ 의 크기는 `normalized_shape` 와 동일하며 초기값은 1입니다.
- bias : 학습 가능한 편향 $\beta$ 의 크기는 `normalized_shape` 와 동일하며 초기값은 0 입니다.


## example code

```python
import torch
import torch.nn as nn

input_tensor = torch.randn(10, 20, 30)

# 마지막 차원에 대해 정규화 적용
layer_norm = nn.LayerNorm(30)

output_tensor = layer_norm(input_tensor)

print(f"Input shape : {input_tensor.shape}")
print(f"Output shape : {output_tensor.shape}")
```

```
Input shape : torch.Size([10, 20, 30])
Output shape : torch.Size([10, 20, 30])
```

```python
import torch
import torch.nn as nn

t = torch.arange(start = 1,end = 31, step=3, dtype=torch.float32)

layer_norm = nn.LayerNorm(t.numel())

print(t)

print(layer_norm(t))

```

```
tensor([ 1.,  4.,  7., 10., 13., 16., 19., 22., 25., 28.])
tensor([-1.5667, -1.2185, -0.8704, -0.5222, -0.1741,  0.1741,  0.5222,  0.8704,
         1.2185,  1.5667], grad_fn=<NativeLayerNormBackward0>)
```

