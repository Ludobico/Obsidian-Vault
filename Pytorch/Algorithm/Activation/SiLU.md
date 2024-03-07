SILU는 sigmoid-weighted Linear Unit의 약어로, 입력 값에 [[Sigmoid]] 함수를 적용한 후 선형 함수를 적용하는 것을 의미합니다.

SILU 함수는 다음과 같이 정의됩니다.

$$silu(x) = x * \sigma(x)$$
$$\sigma(x) = \frac{1}{1+e^{-x}}$$
![[Pasted image 20240304151301.png]]

[[Pytorch]]로 작성된 코드를 보면 다음과 같이 결과가 나오게 됩니다.

```python
import torch
import torch.nn as nn

input_x = torch.randn(8, dtype=torch.float16)

relu_act = nn.ReLU()
silu_act = nn.SiLU()

print(input_x)

relu_output = relu_act(input_x)
silu_output = silu_act(input_x)

print(relu_output)
print(silu_output)
```

```
## input값
tensor([-0.3230, -1.7627, -0.3347,  2.4512,  0.8359,  0.7974, -0.5151, -0.4985],
       dtype=torch.float16)
       
## relu activation 적용
tensor([0.0000, 0.0000, 0.0000, 2.4512, 0.8359, 0.7974, 0.0000, 0.0000],
       dtype=torch.float16)
       
## silu activation 적용
tensor([-0.1356, -0.2581, -0.1396,  2.2559,  0.5830,  0.5498, -0.1926, -0.1884],
       dtype=torch.float16)
```

[[ReLU]] 의 범위인 $[0, \infty]$ 와 다르게 SiLU의 범위는 $[-1, \infty]$ 로 확장됩니다.

