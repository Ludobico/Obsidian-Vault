`torch.rsqrt` 함수는 [[Pytorch]] 에서 사용되는 함수로, **주어진 텐서의 각 요소에 대한 역제곱근(reciprocal of the square root)을 계산**하여 반환합니다.

$$
\text{out}_i = \frac{1}{\sqrt{\text{input}_i}}
$$

> torch.rsqrt(input, \*, out=None) → [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 역제곱근을 계산할 입력 텐서

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 출력 텐서

```python
import torch

# 입력 텐서 생성
a = torch.tensor([4.0, 9.0, 16.0, 25.0])
print(a)

# 역제곱근 계산
rsqrt_a = torch.rsqrt(a)
print(rsqrt_a)
```

```
tensor([ 4.,  9., 16., 25.])
tensor([0.5000, 0.3333, 0.2500, 0.2000])
```
