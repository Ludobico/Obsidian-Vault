`torch.nn.softplus` 는 [[Pytorch]] 에서 제공하는 활성화 함수 중 하나로, **주어진 입력 텐서의 각 요소에 softplus 함수를 적용**합니다. Softplus는 [[ReLU]] 함수의 **부드러운 근사치**로, 출력 값을 항상 **양수로 제한**하는데 사용됩니다.

Softplus 함수는 다음과 같이 정의됩니다.

$$
\text{softplus}(x) = \frac{1}{\beta} * \log(1 + \text{exp}(\beta * x))
$$

여기서 $\beta$ 는 함수의 기울기를 조절하는 파라미터입니다.

```python
torch.nn.Softplus(_beta=1.0_, _threshold=20.0_)
```

> beta -> float
- Softplus 함수의 $\beta$ 값입니다. 기본값은 1입니다.

> threshold -> float
- 입력 값이 임계값을 초과한 경우, 수치적 안정성을 위해 softplus 함수가 선형 함수로 대체됩니다. 기본값은 20입니다.

```python
import torch
import torch.nn as nn

# Softplus 활성화 함수 객체 생성
softplus = nn.Softplus(beta=1.0, threshold=20.0)

# 입력 텐서 생성
input_tensor = torch.tensor([-1.0, 0.0, 1.0, 10.0])

# Softplus 함수를 적용
output_tensor = softplus(input_tensor)

print(output_tensor)
```

```
tensor([ 0.3133,  0.6931,  1.3133, 10.0000])
```

![[Pasted image 20240702143356.png]]