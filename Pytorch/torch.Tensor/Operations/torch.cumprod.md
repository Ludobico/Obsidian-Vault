`torch.cumprod` 는 **텐서의 요소들의 누적 곱을 계산하는 [[Pytorch]] 함수**입니다. 이 함수는 지정된 차원(`dim`)을 다라 텐서의 요소들을 순차적으로 곱합니다.

예를 들어, 입력 텐서가 벡터인 경우, 결과 텐서도 동일한 크기의 벡터가 됩니다. 결과 텐서의 i번째 요소는 입력 텐서의 1번째 요소부터 i번째 요소까지의 곱으로 계산됩니다. 수식으로 표현하면 다음과 같습니다.

$$y_i = x_1 \times x_2 \times x_3 \cdots \times x_i $$

만약 2차원 텐서일 경우 `dim=1` 과 `dim=0` 일때 아래와 같이 계산됩니다.

$$A = \begin{bmatrix}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}$$

$$A_{dim1} = \begin{bmatrix}a_{11} & a_{11} \times a_{12} & a_{11} \times a_{12} \times a_{13} \\ a_{21} & a_{21} \times a_{22} & a_{21} \times a_{22} \times a_{23} \end{bmatrix}$$

$$A_{dim0} = \begin{bmatrix}a_{11} & a_{12} & a_{13} \\ a_{11} \times a_{21} & a_{12} \times a_{22} & a_{13} \times a_{23} \end{bmatrix}$$

> torch.cumprod(_input_, _dim_, _*_, _dtype=None_, _out=None_) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")

이 함수는 다음과 같은 파라미터를 가집니다.

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 입력 텐서

> dim -> int
- 연산을 수행할 차원

> dtype -> [[torch.dtype]] , optional
- 결과 텐서의 데이터 타입을 지정합니다. 지정하지 않으면 입력 텐서의 데이터 타입을 따릅니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 출력 텐서를 지정합니다.

`torch.cumprod` 는 주로 시계열 데이터나 누적 통계량 계산 등에 사용됩니다. 예를 들어, 주식 가격의 누적 수익률을 계산하거나 확률 분포에서 누적 곱을 구하는 데 활용할 수 있습니다.

## example code
---


```python
import torch

tensor1d = torch.tensor([1,2,3,4], dtype=torch.float16)

# 1차원 텐서에 대해 누적 곱 계산
cumprod1d = torch.cumprod(tensor1d, dim=0)
print("1D Cumulative Product:", cumprod1d)

# 2차원 텐서 생성
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 2차원 텐서에 대해 누적 곱을 각 행(row)에 대해 계산
cumprod2d_row = torch.cumprod(tensor2d, dim=1)
print("2D Cumulative Product along rows:", cumprod2d_row)

# 2차원 텐서에 대해 누적 곱을 각 열(column)에 대해 계산
cumprod2d_col = torch.cumprod(tensor2d, dim=0)
print("2D Cumulative Product along columns:", cumprod2d_col)

```

```
1D Cumulative Product: tensor([ 1.,  2.,  6., 24.], dtype=torch.float16)
2D Cumulative Product along rows: tensor([[  1.,   2.,   6.],
        [  4.,  20., 120.]])
2D Cumulative Product along columns: tensor([[ 1.,  2.,  3.],
        [ 4., 10., 18.]])
```

