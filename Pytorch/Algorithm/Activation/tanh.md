하이퍼볼릭탄젠트(Hyperbolic tangent) 함수는 비선형 활성화 함수 중 하나로, [[Pytorch]] 와 같은 딥러닝 프레임워크에서 제공됩니다. 하이퍼볼릭탄젠트 함수는 [[Sigmoid]] 함수의 확장으로 볼 수 있습니다.

수학적으로 하이퍼볼릭탄젠트 함수는 다음과 같이 정의됩니다.

$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

하이퍼볼릭탄젠트 함수는 시그모이드 함수와 유사하지만, <font color="#ffff00">출력 범위가 -1부터 1까지로 확장</font>되어 있습니다. 이는 함수의 출력이 0 주변에서 대칭적이며, 입력값이 큰 경우에도 [[Gradient Vanishing]] 문제가 시그모이드 함수보다 덜 발생한다는 장점을 가지고 있습니다.

![[Pasted image 20240228110224.png]]

```python
import torch
import torch.nn as nn

m = nn.Tanh()
input = torch.randn(3)
print(input)

print("-"*50)

output = m(input)
print(output)
```

```
tensor([ 1.5331,  0.2898, -0.5523])
--------------------------------------------------
tensor([ 0.9109,  0.2819, -0.5023])
```

