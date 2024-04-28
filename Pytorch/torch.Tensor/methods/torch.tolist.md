`.tolist()` 메서드는 [[Pytorch/torch.Tensor/torch.Tensor]] 타입을 <font color="#ffff00">파이썬 리스트로 변환</font>하는 메서드입니다. 이 메서드는 **모든 차원을 리스트로 변환** 합니다.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
y = x.tolist()
print(y)
```

```
[[1, 2], [3, 4]]
```

