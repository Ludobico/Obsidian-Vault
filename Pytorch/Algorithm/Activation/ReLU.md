ReLU 함수는 <font color="#ffff00">입력값이 0보다 작을 때는 0을 출력하고, 그렇지 않으면 입력값을 그대로 출력하는 비선형 활성화 함수</font>입니다. 

이 함수의 정의는 다음과 같습니다.

$$ReLU(x) = (x)^+ = \max(0,x)$$

![[Pasted image 20240228162602.png]]


```python
import torch
import torch.nn as nn

x = torch.randn(4,2)

print(x)

relu_func = nn.ReLU()

y = relu_func(x)

print(y)
```

```
tensor([[-0.2962, -0.5217],
        [-0.5430,  1.5400],
        [ 1.3782,  1.6334],
        [-0.4782,  1.0468]])
tensor([[0.0000, 0.0000],
        [0.0000, 1.5400],
        [1.3782, 1.6334],
        [0.0000, 1.0468]])
```