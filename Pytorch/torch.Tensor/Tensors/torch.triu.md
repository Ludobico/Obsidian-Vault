`torch.triu` 함수는 행렬(2차원 텐서) 또는 행렬 배치의 **상삼각 부분을 반환**합니다. 반환된 텐서에서 나머지 요소들은 0으로 설정됩니다. 행렬의 상삼각 부분은 주대각선 및 그 위의 요소들로 정의됩니다.

> torch.triu(_input_, _diagonal=0_, _out=None_) → [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> inpuut -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 입력 텐서입니다.

> diagonal -> int, optional
- 고려할 대각선을 설정합니다. 기본값은 0입니다.
- diagonal = 0 : 주대각선 및 그 위의 모든 요소들을 포함합니다.
- diagonal > 0 : 주대각선 위의 특정 대각선을 제외합니다.
- diagonal < 0 : 주대각선 아래의 특정 대각선을 포함합니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]], optional
- 결과를 저장할 출력 텐서입니다.

주대각선은 `(i, i)` 인덱스를 가지며, 여기서 `i` 는 `[0, min(d1, d2) -1]` 범위에 속합니다. `d1` 과 `d2` 는 행렬의 차원입니다.

## example code

```python
import torch


x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(x)
print('-'*80)
print(torch.triu(x))
print('-'*80)
print(torch.triu(x, diagonal=1))
print('-'*80)
print(torch.triu(x, diagonal=-1))
```

```
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
--------------------------------------------------------------------------------
tensor([[1, 2, 3],
        [0, 5, 6],
        [0, 0, 9]])
--------------------------------------------------------------------------------
tensor([[0, 2, 3],
        [0, 0, 6],
        [0, 0, 0]])
--------------------------------------------------------------------------------
tensor([[1, 2, 3],
        [4, 5, 6],
        [0, 8, 9]])
```

