
`torch.clamp` 는 [[Pytorch]] 에서 제공하는 함수로, **입력 텐서의 각 요소를 특정 범위 내로 제한(clamp)하는데 사용**됩니다. 이 함수는 주어진 최소값(min) 과 최대값(max)을 기준으로, 입력 값이 이 범위를 벗어나지 않도록 조정합니다.

```python
torch.clamp(input, min=None, max=None, *, out=None) → Tensor
```

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 클램프할 입력 텐서입니다.

> min -> Number or [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 클램프할 범위의 하한값입니다. 이 값보다 작은 값은 모두 이 하한값으로 설정됩니다.

> max -> Number or [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 클램프할 범위의 상한값입니다. 이 값보다 큰 값은 모두 이 상한값으로 설정됩니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]], optional
- 결과를 저장할 출력 텐서입니다.

`torch.clamp` 는 입력 텐서의 각 요소 $x_i$ 에 대해서 다음과 같이 동작합니다.

$$
y_i = \min(\max(x_i, minv_i), maxv_i)
$$

```python
import torch

input_tensor = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])

min_value = 1.0
max_value = 2.0

clamped_tensor = torch.clamp(input_tensor, min=min_value, max=max_value)

print(clamped_tensor)
```

```
tensor([1.0000, 1.5000, 2.0000, 2.0000, 2.0000])
```

