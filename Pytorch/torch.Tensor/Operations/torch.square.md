`torch.square` 는 [[Pytorch]] 에서 제공하는 함수로, 주어진 입력 텐서의 **각 요소를 제곱한 새로운 텐서를 반환**합니다.

```python
torch.square(input, *, out=None) → Tensor
```

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 제곱할 입력 텐서입니다.

> out  -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]], optional
- 입력 텐서의 각 요소를 제곱한 결과를 포함하는 새로운 텐서입니다.

```python
import torch

# 임의의 입력 텐서 생성
a = torch.tensor([1,2,3,4])
print("입력 텐서 a:", a)

# 입력 텐서의 각 요소를 제곱한 새로운 텐서 생성
squared_tensor = torch.square(a)
print("제곱된 텐서:", squared_tensor)
```

```
입력 텐서 a: tensor([1, 2, 3, 4])
제곱된 텐서: tensor([ 1,  4,  9, 16])
```

