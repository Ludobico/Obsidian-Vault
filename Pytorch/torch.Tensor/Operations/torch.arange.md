`torch.arange` 는 [[Pytorch]] 에서 **지정된 범위와 간격으로 1차원 텐서를 생성**하는 함수입니다. 이 함수는 NumPy의 `arange` 와 유사하며, 지정된 시작 값부터 끝 값까지 일정한 간격으로 값을 생성하빈다.

```python
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

> start -> Default 0
- 생성할 값의 시작 값입니다.

> end
- 생성할 값의 끝 값입니다. 이 값은 포함되지않으므로, \[start, end\) 범위의 값을 생성합니다.

> step -> Default 1
- 각 값 사이의 간격입니다.

> out  -> optional
- 결과를 저장할 텐서입니다.

> dtype -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , Default torch.float32, optional
- 반환되는 텐서의 데이터 타입입니다.

> layout -> torch.layout, optional
- 반환되는 텐서의 레이아웃입니다. 기본값은 `torch.strided` 입니다.

> device -> torch.device, optional
- 반환되는 텐서가 저장될 장치입니다. 기본값은 현재 설정된 장치입니다.

> requires_grad -> bool, optional
- 반환되는 텐서에게 autograd가 연산을 기록할지 여부를 지정합니다. 기본값은 `False` 입니다.

## Note
---
- 부동 소수점 간격 : 부동 소수점 간격을 사용하는 경우, 부동 소수점 연산의 특성으로 인해 비교시소수점 오차가 발생할 수 있습니다. 이를 피하기 위해 `end` 값을 약간 감소시킬 수 있습니다.

- 출력 텐서 지정 : `out` 파라미터를 사용하여 결과를 기존 텐서에 저장할 수 있습니다.

- Autograd 사용 : `requires_grad` 를 `True`로 설정하면 생성된 텐서에서 수행된 연산이 자동미분을 위해 기록됩니다.

## Example code
---

```python
import torch

tensor1 = torch.arange(10)
print(tensor1)

tensor2 = torch.arange(2,8)
print(tensor2)

tensor3 = torch.arange(1, 10, 2)
print(tensor3)

tensor4 = torch.arange(0,1 ,0.2)
print(tensor4)

tensor5 = torch.arange(0, 5, dtype=torch.float32, device='cuda')
print(tensor5)
```

```
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([2, 3, 4, 5, 6, 7])
tensor([1, 3, 5, 7, 9])
tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])
tensor([0., 1., 2., 3., 4.], device='cuda:0')
```

