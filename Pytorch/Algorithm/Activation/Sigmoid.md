시금이드 활성화 함수는 신경망에서 주로 사용되는 활성화 함수 중 하나입니다. 주로 로지스틱 함수로 알려져있으며, 이 함수는 <font color="#ffff00">입력값을 0과 1 사이의 값으로 변환</font>해주는데, 이 특성 때문에 주로 이진 분류 문제에서 출력층의 활성화 함수로 사용됩니다. 시그모이드 함수는 다음과 같이 표현됩니다.

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

![[Pasted image 20240228105153.png]]

```python
import torch
import torch.nn as nn

m = nn.Sigmoid()
input = torch.randn(2)
print(input)

print("-"*50)

output = m(input)
print(output)
```

```
tensor([ 0.9888, -2.0415])
--------------------------------------------------
tensor([0.7288, 0.1149])
```
