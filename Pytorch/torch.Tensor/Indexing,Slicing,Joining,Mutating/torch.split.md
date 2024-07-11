`torch.split` 함수는 [[Pytorch]] 에서 사용되는 함수로, 주어진 텐서를 **특정 크기 또는 섹션으로 분할**합니다. 이 함수는 원래 텐서의 뷰([[torch.view]]) 를 반환하므로, 각 분할된 텐서는 원래 텐서의 데이터를 공유합니다. 결과값으로 <font color="#ffff00">분할된 텐서들의 튜플</font>을 반환합니다.

```python
torch.split(tensor, split_size_or_sections, dim=0) → Tuple[Tensor, …]
```

> tensor -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 분할할 입력 텐서입니다.

> split_size_or_sections -> int or list[int]
- 분할 크기 또는 각 섹션의 크기를 지정하는 리스트입니다.
	- 정수로 주어지면, 텐서는 지정된 크기만큼 균등하게 분할됩니다. 만약 주어진 차원(dim)의 크기가  `split_size` 로 나누어 떨어지지 않으면, 마지막에는 더 작을 수 있습니다.
	- 리스트로 주어지면, 리스트의 각 요소는 분할된 각 섹션의 크기를 나타냅니다.

> dim -> int
- 텐서를 분할할 차원을 지정합니다. 기본값은 0입니다.


## example code

```python
import torch

tensor = torch.arange(10)
print(tensor)

split_tensors = torch.split(tensor, 3)
print(split_tensors)
```

```
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
(tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8]), tensor([9]))
```

```python
import torch

tensor = torch.arange(10)
print(tensor)

split_tensors = torch.split(tensor, [3,3,4])
print(split_tensors)
```

```
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
(tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8, 9]))
```

