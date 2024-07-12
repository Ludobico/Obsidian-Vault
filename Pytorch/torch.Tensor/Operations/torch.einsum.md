`torch.einsum` 은 [[Pytorch]] 에서 Einstein summation convention을 기반으로 한 텐서 연산을 수행하기 위한 함수입니다. 이 함수는 여러 차원의 텐서를 다룰 때 사용되며, **다양한 선혀 대수 연산을 간결한 형식으로 표현할 수 있게 해줍니다.**

## Einstein Summation Convention
- ESC 는 다차원 배열에서의 텐서 곱셈, 텐서 내적, 축소(summation), 행렬 곱셈 등을 간결하게 표현하는 방법입니다.
- 텐서의 차원 레벨(subscript)을 부여하고, 이를 사용하여 연산을 정의합니다.

> torch.einsum(_equation_, _*operands_)  -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> equation
- 서브스크립트는 \[a-zA-Z\] 사이의 문자로 표현되며, 입력 텐서의 차원 순서와 동일하게 정렬됩니
- ','를 사용하여 각 입력 텐서의 서브스크립트를 구분합니다.
- '->' 를 사용하여 출력 텐서의 서브스크립트를 명시적으로 정의할 수 있습니다.

## example code

```python
import torch

A = torch.tensor([[1,2], [3,4]])
B = torch.tensor([[5,6], [7,8]])

# 행렬 곱셈
C = torch.einsum('ij,jk->ik', A, B)
print(C)

```

```
tensor([[19, 22],
        [43, 50]])
```

```python
import torch

batch_A = torch.randn(2,3,4)
batch_B = torch.randn(2,4,5)
print(batch_A.size())
print(batch_B.size())

# 배치 행렬 곱셈
batch_C = torch.einsum("bnm,bmp -> bnp", batch_A, batch_B)
print(batch_C.size())
```

```
torch.Size([2, 3, 4])
torch.Size([2, 4, 5])
torch.Size([2, 3, 5])
```

