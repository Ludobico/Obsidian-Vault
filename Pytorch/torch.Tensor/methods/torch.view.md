```python
torch.view(*shape) -> torch.Tensor
```

`torch.view()` 는 텐서의 데이터를 그대로 유지하면서 모양(shape)를 변경하는데 사용됩니다. 새로운 모양의 텐서를 반환하지만 원본 데이터를 공유합니다. 다만, 두 텐서의 요소 수는 동일해야 합니다.

> shape -> torch.Size or int

```python
import torch


x = torch.randn(4, 4)
print(x.shape)

y = x.view(16)
print(y.shape)

z = x.view(-1, 8)
print(z.shape)

```

```
torch.Size([4, 4])

torch.Size([16])

torch.Size([2, 8])
```
