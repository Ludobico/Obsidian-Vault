`torch.cumsum` 은 [[Pytorch]] 에서 사용되는 함수로, 주어진 텐서의 **요소 들의 누적 합을 계산하여 반환**합니다. 이 함수는 지정된 차원(`dim`) 을 따라 누적 합을 계산합니다.

$$
y_i = x_1 + x_2 + x_3 + \dots + x_i
$$

> torch.cumsum(_input_, _dim_, _*_, _dtype=None_, _out=None_) → [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 입력 텐서입니다.

> dim -> int
- 누적 합을 계산할 차원입니다.

> dtype -> [[torch.dtype]] , optional
- 반환되는 텐서의 원하는 데이터 타입입니다. 지정된다면, 입력 텐서는 연산이 수행되기 전에 해다ㅇ 데이터 타입으로 변환됩니다. 이는 데이터 타입 오버플로우를 방지하는데 사용됩니다. 기본값은 `None` 입니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]], optional
- 출력 텐서입니다.

## example code

```python
import torch

# 1차원 텐서에서 누적 합 계산
tensor = torch.tensor([1, 2, 3, 4])
cumsum_tensor = torch.cumsum(tensor, dim=0)
print(cumsum_tensor)


# 2차원 텐서에서 차원 0을 따라 누적 합 계산
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
cumsum_tensor_2d_dim0 = torch.cumsum(tensor_2d, dim=0)
print(cumsum_tensor_2d_dim0)


# 2차원 텐서에서 차원 1을 따라 누적 합 계산
cumsum_tensor_2d_dim1 = torch.cumsum(tensor_2d, dim=1)
print(cumsum_tensor_2d_dim1)

```

```
tensor([ 1,  3,  6, 10])
tensor([[ 1,  2,  3],
        [ 5,  7,  9],
        [12, 15, 18]])
tensor([[ 1,  3,  6],
        [ 4,  9, 15],
        [ 7, 15, 24]])
```

