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

