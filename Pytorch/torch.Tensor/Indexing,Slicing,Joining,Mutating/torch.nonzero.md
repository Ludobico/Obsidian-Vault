`torch.nonzero` 함수는 [[Pytorch]] 에서 사용되는 함수로, **주어진 텐서에서 0이 아닌 값들의 인덱스를 반환**합니다. 이 함수는 두 가지 방식으로 동작할 수 있습니다. `as_tuple` 이`False` 인 기본 설정 방식과 `as_tuple` 이 `True` 인 방식입니다.

> torch.nonzero(_input_, _*_, _out=None_, _as_tuple=False_) → LongTensor or tuple of LongTensors
## as_tuple = False

기본 설정인 `as_tuple = False` 로 호출할 경우, 함수는 2차원 텐서를 반환합니다. 이 텐서의 각 행은 입력 텐서에서 0이 아닌 값의 인덱스를 나타냅니다. 결과 텐서는 사전식 순서로 정렬되며, 마지막 인덱스가 가장 빠르게 변합니다.

예를 들어, 입력 텐서가 $n$ 차원이라면, 반환되는 텐서의 크기는 ($z \times n$)  이 됩니다. 여기서 $z$ 는 입력 텐서에서 0이 아닌 값의 총 개수입니다.

## as_tuple = True

`as_tuple = True` 로 호출할 경우, 함수는 1차원 텐서들의 튜플을 반환합니다. 각 텐서는 입력 텐서의 각 차원에 대한 0이 아닌 값의 인덱스를 포함합니다.

예를 들어, 입력 텐서가 $n$ 차원이라면, 반환되는 튜플은 $n$ 개의 1차원 텐서를 포함하여, 각 텐서의 크기는 $z$ 입니다. 여기서 $z$ 는 입력 텐서에서 0이 아닌 값의 총 개수입니다.

## example code

```python
import torch

tensor = torch.tensor([[1,0,0], [0, 1, 1], [0, 0, 1]])

nonzero_indices = torch.nonzero(tensor)
print(nonzero_indices)

print('-'*80)
print('-'*80)
print('-'*80)

nonzero_indices_tuple = torch.nonzero(tensor, as_tuple=True)
print(nonzero_indices_tuple)
```

```
tensor([[0, 0],
        [1, 1],
        [1, 2],
        [2, 2]])
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
(tensor([0, 1, 1, 2]), tensor([0, 1, 2, 2]))
```

