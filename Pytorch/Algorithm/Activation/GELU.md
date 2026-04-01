GELU(Gaussian Error Linear Unit) 함수는 비선형 활성화 함수 중 하나로, [[Pytorch]]와 같은 딥러닝 프레임워크에서 제공됩니다. BERT, GPT 등 최신 트랜스포머 기반의 모델들에서 표준으로 사용되는 활성화 함수입니다.

수학적으로 GELU 함수는 표준 정규 분포의 누적 분포 함수인 $\phi(x)$ 를 사용하여 다음과 같이 정의됩니다.

$$
\text{GELU}(x) = x \phi(x)
$$

이는 계산의 편의를 위해 [[tanh]] 등을 사용하여 다음과 같이 근사하여 계산하기도 합니다.

$$
\text{GELU}(x) \approx 0.5 x \cdot (1 + tanh(\sqrt{\frac{2}{\pi}}  \cdot (x + 0.044715 x^3)))
$$

GELU 함수는 [[ReLU]]와 유사하게 양수 입력에 대해서는 선형적으로 증가하지만, 음수 입력에 대해서는 즉각적으로 0이 되지 않고 부드러운 곡선을 그리며 약간의 음수 값을 허용합니다. 이러한 매끄러운 곡선 형태는 미분 가능성을 보장하고, 학습 중 뉴런이 죽어버리는 Dying ReLU 문제를 완화하여 복잡한 신경망 모델에서 더 높은 성능을 이끌어냅니다.

```python
import torch
import torch.nn as nn

m = nn.GELU()
inputs = torch.tensor([ 1.5000,  0.5000, -1.0000])
print(inputs)
print("-" * 50)
output = m(inputs)
print(output)
```

```
tensor([ 1.5000,  0.5000, -1.0000])
--------------------------------------------------
tensor([ 1.3998,  0.3457, -0.1587])
```

