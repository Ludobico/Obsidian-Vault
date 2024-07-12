`torch.masked_fill` 메서드는 [[Pytorch]] 에서 텐서의 특정 위치에서 **마스크(mask)에 따라 값을 채우는 역할**을 합니다. 이 메서드는 주어진 마스크가 True인 위치에 해당하는 원소들에 대해 지정된 값을 채워 넣습니다.

> Tensor.masked_fill (mask, value) -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

> mask -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 마스크 텐서로, 기존 텐서와 동일한 크기([[torch.shape]])를 가져야합니다. 이 마스크는 `True` 또는 `False` 값을 가지는 텐서여야 하며, `True` 인 위치의 원소에만 값을 채워 넣습니다.

> value -> [[torch.dtype]] value
- 주어진 마스크가 `True` 인 위치에 해당하는 원소들에 대해 원하는 값을 채워 넣습니다. 나머지 원소들은 변경되지 않습니다.

```python
import torch

tensor = torch.tensor([[1,2,3],
                       [4,5,6]], dtype=torch.int16)
mask = torch.tensor([[True, False, True],
                     [False, False, True]])

tensor = tensor.masked_fill(mask, -512)
print(tensor)
```

```
tensor([[-512,    2, -512],
        [   4,    5, -512]], dtype=torch.int16)
```

$$
\frac{QK^T}{\sqrt{d_k}}V
$$