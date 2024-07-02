
`torch.erf` 는 [[Pytorch]] 에서 제공하는 함수로, 주어진 입력 텐서의 각 요소에 대해 **오차 함수(error function)를 계산**합니다. 오차 함수는 다음과 같이 정의됩니다.

$$
erf(x) = \frac{2}{\sqrt{\pi}}
\int_0^x \mathrm{e}^{-t^2}\,\mathrm{dt}

$$

이 함수는 수치적 계산에서 중요한 역할을 하며, 주로 확률 이론, 통계학, 물리학, 공학 등의 분야에서 사용됩니다.

`torch.erf` 는 다음과 같은 파라미터를 받습니다.

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 오차 함수를 계산할 입력 텐서입니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 출력 텐서입니다.

```python
import torch

# 입력 텐서를 생성합니다.
input_tensor = torch.tensor([0, -1., 10.])

# torch.erf 함수를 사용하여 입력 텐서의 각 요소에 대해 오차 함수를 계산합니다.
result_tensor = torch.erf(input_tensor)

# 결과를 출력합니다.
print(result_tensor)
```

```
tensor([ 0.0000, -0.8427,  1.0000])
```

