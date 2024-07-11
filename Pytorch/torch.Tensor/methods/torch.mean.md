
## Average of All Tensor Elements

이 버전은 **입력 텐서의 모든 요소의 평균을 계산**합니다. 입력 텐서는 부동 소수점 또는 복소수형이어야합니다. 계산이 끝나면 결과값으로 모든 요소의 평균 값을 포함하는 <font color="#ffff00">스칼라 텐서</font>를 반환합니다.

> torch.mean(input, \*, dtype=None) → [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 부동 소수점 또는 복소수형 텐서

> dtype -> [[torch.dtype]], optional
- 반환될 텐서의 원하는 데이터 타입, 지정된 경우 연산 전에 입력 텐서가 아닌 이 데이터 타입으로 변환됩니다. 기본값은 `None` 입니다.


```python
import torch

a = torch.linspace(10, 30, steps=10, dtype=torch.float16)
print(a)

mean_value = torch.mean(a)
print(mean_value)
```

```
tensor([10.0000, 12.2188, 14.4453, 16.6719, 18.8906, 21.1094, 23.3281, 25.5625,
        27.7812, 30.0000], dtype=torch.float16)
tensor(20., dtype=torch.float16)
```

## Average Along Specific dimensions

이 버전은 주어진 **차원 또는 차원들에 대한 평균을 계산**합니다.

> torch.mean(input, dim, keepdim=False, \*, dtype=None, out=None) → [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 입력 텐서

> dim -> int, tuple
- 평균을 계산할 차원

> keepdim -> bool, Default : False
- 출력 텐서가 입력 텐서와 동일한 차원을 유지할지 여부이며 False일시 지정된 차원은 축소됩니다.

> dtype -> [[torch.dtype]], optional
- 반환될 텐서의 원하는 데이터 타입이며 지정된 경우 여산 전에 입력 텐서가 이 데이터 타입으로 변환됩니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]], optional
- 출력 텐서

```python
import torch

a = torch.arange(1, 13, dtype=torch.float16).reshape(3,4)
print(a)

mean_value_dim0 = torch.mean(a, dim=0)
print(mean_value_dim0)
mean_value_dim0_keep = torch.mean(a, dim=0, keepdim=True)
print(mean_value_dim0_keep)


mean_value_dim1 = torch.mean(a, dim=1)
print(mean_value_dim1)
mean_value_dim1_keep = torch.mean(a, dim=1, keepdim=True)
print(mean_value_dim1_keep)
```

```
tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  8.],
        [ 9., 10., 11., 12.]], dtype=torch.float16)
tensor([5., 6., 7., 8.], dtype=torch.float16)
tensor([[5., 6., 7., 8.]], dtype=torch.float16)
tensor([ 2.5000,  6.5000, 10.5000], dtype=torch.float16)
tensor([[ 2.5000],
        [ 6.5000],
        [10.5000]], dtype=torch.float16)
```

