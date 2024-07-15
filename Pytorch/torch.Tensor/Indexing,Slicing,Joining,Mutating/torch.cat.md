```python
torch.cat(tensors, dim=0, out=None)
```

`.cat()` 함수는 여러 개의 텐서를 <font color="#ffff00">지정한 차원을 기준으로 연결하여 새로운 텐서를 생성</font>합니다. 

> tensors -> list or tuple
- 연결할 텐서를 담고 있는 리스트나 튜플입니다.

> dim -> int
- 연결할 차원을 지정합니다. 연결할 차원을 0부터 시작하여 텐서의 차원 수보다 작은 +정수로 지정합니다. 기본값은 0으로, 첫 번째 차원(행)을 기준으로 텐서를 연결합니다.

> out
- 결과를 저장할 출력 텐서입니다. 이 매개변수를 사용하여 결과를 직접 지정할 수 있습니다. 기본값은 `None` 이며, 새로운 텐서가 생성됩니다.

예를 들어, 두 개의 2차원 텐서를 행(0차원)을 기준으로 연결하는 경우 다음과 같이 사용할 수 있습니다.

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

result = torch.cat((tensor1, tensor2), dim=0)
print(result)
```

```
tensor([[1, 2],
[3, 4],
[5, 6],
[7, 8]])
```

텐서를 연결할때, 연결하려는 텐서들의 <font color="#ffff00">차원이 서로 일치하지 않는 경우 오류가 발생</font>합니다. 

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])  # 2차원 텐서
tensor2 = torch.tensor([[[5, 6], [7, 8]]])  # 3차원 텐서

result = torch.cat((tensor1, tensor2), dim=0)
```

```
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
```

`torch.cat()` 을 사용할때, **지정된 차원을 제외한 다른 차원의 크기는 동일**해야 작동합니다.

```python
t1 = torch.randn(2,3,1,2,3)
t2 = torch.randn(2,25,1,2,3)

tc = torch.cat((t1, t2), dim=1)

print(tc.shape)
```

```
torch.Size([2, 28, 1, 2, 3])
```

```python
import torch
import torch.nn as nn


t1 = torch.randn(2,3,1,2,3)
t2 = torch.randn(2,25,1,4,3)

tc = torch.cat((t1, t2), dim=1)

print(tc.shape)
```

```
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 2 but got size 4 for tensor number 1 in the list.
```

