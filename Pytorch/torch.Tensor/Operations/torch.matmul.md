`torch.matmul` 은 [[Pytorch]] 에서 **다양한 차원의 텐서들 간의 행렬 곱셈을 수행하는 함수**입니다. 입력 텐서들의 차원에 따라 적절한 행렬 연산을 수행합니다. 다음은 torch.matmul의 동작에 대한 상세한 설명입니다.

### 1-D Tensor -> Vector
- `input` 과 `other` 모두 1차원 텐서일 경우, dot product를 수행하여 스칼라 값을 반환합니다.

### 2-D Tensor -> Matrix
- `input` 과 `other` 모두 2차원 텐서일 경우, 일반적인 행렬-행렬 곱셈을 수행합니다.


이 함수는 다음과 같은 파라미터를 가집니다.

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 곱셈을 수행할 첫번째 텐서입니다.

> other -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 곱셈을 수행할 두번째 텐서입니다.

> result -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , Optional
- 결과를 저장할 출력 텐서입니다.

```python
import torch

tensor1 = torch.randn(3)
tensor2 = torch.randn(3)

result = torch.matmul(tensor1, tensor2)

print(tensor1)
print(tensor2)
print(result)
```

```
tensor([ 1.0776,  0.7050, -0.9283])
tensor([-1.4010, -0.0434,  0.6756])
tensor(-2.1674)
```

```python
import torch

# 2차원 행렬-행렬 곱셈
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(a, b)
print(result)  # 출력: tensor([[19, 22], [43, 50]])
```

```
tensor([[19, 22],
        [43, 50]])
```

```python
import torch

# 1차원과 2차원 곱셈
a = torch.tensor([1, 2])
b = torch.tensor([[3, 4], [5, 6]])
result = torch.matmul(a, b)
print(result)  # 출력: tensor([13, 16])
```

```
tensor([13, 16])
```

