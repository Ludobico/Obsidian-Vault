`.dim()` 메서드는 <font color="#ffff00">텐서의 차원 수(랭크)를 반환</font>합니다. 예를 들어, 1D 텐서의 경우 1을 반환하고 2D 텐서의 경우 2를 반환합니다.

```python
import torch

# 1D tensor
a = torch.tensor([1, 2, 3])
print(a.dim())
```

```
1
```

```python
import torch

# 2D tensor
b = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(b.dim())
```

```
2
```

