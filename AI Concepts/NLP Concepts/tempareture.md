## temparature
---

[[Transformer]] 아키텍처의 최종 output 은 보통 [[Softmax]] 함수를 사용하여 <font color="#ffff00">다음 단어의 확률 분포를 계산하고, 이를 기반으로 다음 단어를 선택</font>합니다. 여기에 tempareture 파라미터를 적용하면 <font color="#ffff00">확률 분포를 완만하게 만듭니다. 이렇게 하여 예측된 단어의 다양성을 높일 수 있습니다. </font>

![[Pasted image 20240304112626.png]]

예를 들어, 다음과 같은 `input_x` 값이 있다고 가정하겠습니다.

```python
import torch

input_x = torch.randn([5], dtype=torch.float16)
```

```
tensor([ 1.6641,  0.9614, -0.2781,  2.5703, -0.6387], dtype=torch.float16)
```

위의 텐서값들을 [[Softmax]] 함수를 통하면 아래의 계산식에 따라 결과가 나오게 됩니다.

$$\frac{e^{x_i}}{\Sigma_{j=1}^{n}e^{x_j}}$$

$$\frac{e^{1.6641}}{e^{1.6641} + {e^{0.9614} + e^{-0.2781}+e^{2.5703}+e^{-0.6387}}}$$
$e^{x_1} = 1.6641, e= \lim_{x \rightarrow \infty}(1+\frac{1}{x})^x$

```python
softmax = nn.Softmax(dim=0)

print(softmax(input_x))
```

```
tensor([0.2373, 0.1176, 0.0340, 0.5874, 0.0237], dtype=torch.float16)
```

여기에<font color="#00b050"> temparature</font>를 적용하면 각 계산식의 분모에 temparature 파라미터가 추가로 적용됩니다.

$$\frac{e^{\frac{x_i}{T}}}{\Sigma_{j=1}^{n}e^{\frac{x_i}{T}}} T=temparature$$

기존의 softmax를 통한 값이
```
tensor([0.2373, 0.1176, 0.0340, 0.5874, 0.0237], dtype=torch.float16)
```

위와 같다면, temparature가 적용된 식은 아래의 식을 통해 확률분포가 변경됩니다.

```
tensor([0.2144, 0.0891, 0.0189, 0.6655, 0.0121])
```

최종 softmax와 temperature를 비교하면 아래와 같이 됩니다.

```python
import torch
import torch.nn as nn

input_x = torch.tensor([1.6641,  0.9614, -0.2781,  2.5703, -0.6387])

print(input_x)


softmax = nn.Softmax(dim=0)


def temparature(input_tensor, temp=0.8):
  scaled_tensor = input_tensor / temp
  return torch.exp(scaled_tensor) / torch.sum(torch.exp(scaled_tensor))


temp_output = temparature(input_x)
print("softmax_output : {0}".format(softmax(input_x)))
print("-"*80)
print("sofrmax_sum : {0}".format(torch.sum(softmax(input_x))))
print("-"*80)
print("temperature_output : {0}".format(temp_output))
print("-"*80)
print("temperature sum output : {0}".format(torch.sum(temp_output)))
```

```
tensor([ 1.6641,  0.9614, -0.2781,  2.5703, -0.6387])
softmax_output : tensor([0.2373, 0.1175, 0.0340, 0.5874, 0.0237])
--------------------------------------------------------------------------------
sofrmax_sum : 1.0
--------------------------------------------------------------------------------
temperature_output : tensor([0.2144, 0.0891, 0.0189, 0.6655, 0.0121])
--------------------------------------------------------------------------------
temperature sum output : 1.0
```



