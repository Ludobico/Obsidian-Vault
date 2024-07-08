오류 함수(Error function, erf)는 수학과 통계학에서 사용되는 특수 함수로, 주로 정규 분포와 관련된 계산에서 사용됩니다. 오류 함수는 다음과 같이 정의됩니다.

$$
erf(x) = \frac{2}{\sqrt{\pi}}
\int_0^x \mathrm{e}^{-t^2}\,\mathrm{dt}
$$

이 함수는 정규 분포의 누적 분포 함수(CDF)의 일부로 사용됩니다. 구체적으로, 오류 함수는 **정규 분포에서 특정 값 이하의 확률을 계산**하는데 도움이 됩니다.

이 함수는 입력값 $x$ 에 대해서 **-1과 1 사이의 값을 출력**합니다.

## error function using Pytorch
---
[[Pytorch]] 에서 오류 함수는 [[torch.erf]] 로 제공됩니다. 이 함수는 주어진 텐서의 각 요소에 대해 오류 함수 값을 계산합니다.

```python
import torch

x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])

y = torch.erf(x)

print("Input Tensor : ", x)

print("Erf Output : ", y)

```

```
Input Tensor :  tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000])
Erf Output :  tensor([0.0000, 0.5205, 0.8427, 0.9661, 0.9953])
```

오류 함수의 그래프를 그려보면, 값이 0일 때 오류 함수의 출력은 0이고, 값이 증가할수록 오류 함수의 출력이 1에 가까워집니다. 이를 시각화하기 위해 maplitlib를 사용할 수 있습니다.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-3, 3, 100)
y = torch.erf(x)

plt.plot(x.numpy(), y.numpy())
plt.title("Error Function (ERF)")
plt.xlabel("x")
plt.ylabel("erf(x)")
plt.grid(True)
plt.show()
```

![[Pasted image 20240708110514.png]]

## Difference between erf and tanh

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-3, 3, 100)
erf_y = torch.erf(x)
tanh_y = torch.tanh(x)

plt.plot(x.numpy(), erf_y.numpy(), label="erf(x)")
plt.plot(x.numpy(), tanh_y.numpy(), label="tanh(x)")
plt.title("Error Function (erf) vs Hyperbolic Tangent Function (tanh)")
plt.xlabel("x")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
```

![[Pasted image 20240708112236.png]]

- erf는 적분 형태로 정의되며, 정규 분포와 관련된 확률 계산에 사용됩니다.
- tanh는 지수 함수로 정의되며, 신경망의 [[Activation]] 으로 사용됩니다.
---
- erf의 기울기는 중간 구간에서 점진적으로 변화합니다.
- tanh의 기울기는 중간 구간에서 더 급격하게 변화하며, 이는 시그모이드 함수와 유사하지만 출력 범위가 -1에서 1로 확장된 것입니다.

---
- erf는 통계적 계산, 정규 분포의 누적 분포 함수(CDF) 계산 등에 사용됩니다.
- tanh는 신경망의 활성화 함수로 사용되어 뉴런의 출력을 -1에서 1 사이로 제한됩니다.

