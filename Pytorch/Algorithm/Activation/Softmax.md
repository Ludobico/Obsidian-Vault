소프트맥스(softmax) 함수는 <font color="#ffff00">입력된 값들을 0과 1 사이의 값으로 변환하여 전체 합이 1이 되도록 만드는 함수</font>입니다. 이는 일종의 확률 분포로 해석할 수 있습니다.

$$Softmax(x) = \frac{e^{x_i}}{\Sigma_{k=1}^Ke^{x_k}}$$
예를 들어 $x = \{1.1, 2.2, 0.2, -1.2\}$ 일때 softmax함수 $s(x)$에 대하여 $s(1.1)$ 은 다음과 같이 구할 수 있습니다.

$$\frac{e^{1.1}}{e^{1.1}+e^{2.2}+e^{0.2}+e^{-1.2}} = 0.22$$

```python
import torch
import torch.nn as nn

x = torch.tensor([1.1, 2.2, 0.2, -1.2])
temp = 0
softsum = 0

for i in x:
  y = torch.exp(i)
  temp += y.item()

for i in x:
  softmax = torch.exp(i).item() / temp
  softsum += softmax
  print(softmax)

print(softsum)
```

```
0.2216806050674722
0.6659653505355732
0.0901286055586732
0.02222543883828143
1.0
```

[[Pytorch]] 로는 다음과 같이 나타낼 수 있습니다.

```python
import torch
import torch.nn as nn

x = torch.tensor([1.1, 2.2, 0.2, -1.2])
softmax_func = nn.Softmax()
softmax_output = softmax_func(x)

print(softmax_output)
print(torch.sum(softmax_output))
```

```
tensor([0.2217, 0.6660, 0.0901, 0.0222])
tensor(1.0000)
```

