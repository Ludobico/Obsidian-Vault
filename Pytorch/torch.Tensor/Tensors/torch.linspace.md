`torch.linspace` 는 `start` 에서 `end` 까지의 범위를 `steps - 1` 로 나누어서 **일정한 간격의 값들을 생성**합니다. 예를 들어, `torch.linspace(3, 10, steps = 5)` 는 3에서 시작하여 10에서 끝나는 5개의 점을 생성하고, 이 점들은 \[3.0, 4.75, 6.5, 8.25, 10.0\] 과 같습니다.

수식으로 표현하면 다음과 같습니다.

$$(start, start+\frac{end - start}{steps - 1}, \dots, start + (steps -2) \times \frac{end - start}{teps - 1}, end)$$

> start -> float or [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 텐서에서 생성할 값의 범위 시작점입니다. 이 파라미터는 부동소수점 숫자나 텐서일 수 있습니다. 텐서를 사용할 경우, 0차원 텐서. 즉, 단일  값을 가지는 텐서여야 합니다.

> end -> float or [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 텐서에서 생성할 값의 범위 끝점입니다. `start` 파라미터와 동일하게 부동소수점 숫자나 0차원 텐서가 될 수 있습니다.

> steps -> int
- 생성할 텐서의 크기입니다. 이 값은 생성될 텐서에 포함될 요소의 수를 의미하며, 결과적으로 이 값의 범위(`start` 에서 `end`) 를 얼마나 많은 단계로 나눌지 결정합니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 텐서를 지정할 수 있습니다. 이 인자를 사용하면 계산 결과를 이 텐서에 직접 저장하여 추가 메모리 할당을 방지할 수 있습니다.

> dtype -> [[torch.dtype]], optional
- 연산을 수행할 데이터 타입을 지정할 수 있습니다. 이 인자를 설정하지 않으면, 전역 기본 데이터 타입이 사용됩니다. 시작점과 끝점이 실수일때는 해당하는 실수형 데이터 타입이, 복소수일 경우에는 해당하는 복소수형 데이터 타입이 자동으로 선택됩니다.

> layout -> torch.layout, optional
- 반환되는 텐서의 메모리 레이아웃을 지정합니다. 기본값은 `torch.strided` 로 이는 메모리 내에 연속적으로 배치된 일반적인 텐서 형태를 의미합니다.

> device -> torch.device, optional
- 반환되는 텐서가 위치할 디바이스를 지정합니다. 디바이스를 지정하지 않으면, 현재 설정된 기본 디바이스(`torch.set_default_device()`) 가 사용됩니다. 예를 들어, CPU를 사용하면 CPU가 기본 디바이스, CUDA를 사용할 경우 현재 CUDA 디바이스가 사용됩니다.

> requires_grad -> bool, optional, Default : False
- 이 파라미터를 `True` 로 설정하면, [[Pytorch]] 의 자동 미분 엔진이 이 텐서에서 수행되는 모든 연산을 추적하게 됩니다. 기본값은 `False` 이며, 대부분의 경우 수치 계산에서 미분값이 필요하지 않을 때 사용됩니다.

## example code
---

```python
import torch

# 3에서 10까지 5단계로 나누어 텐서 생성
tensor = torch.linspace(3, 10, steps=5)
print(tensor) 

# -10에서 10까지 5단계로 나누어 텐서 생성
tensor = torch.linspace(-10, 10, steps=5)
print(tensor) 

# 시작점이 -10, 끝점이 10이며 단계 수가 1인 경우
tensor = torch.linspace(start=-10, end=10, steps=1)
print(tensor)
```

```
tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])
tensor([-10.,  -5.,   0.,   5.,  10.])
tensor([-10.])
```

