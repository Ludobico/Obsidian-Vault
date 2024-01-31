`.item()` 메서드는 스칼라 값으로 표현 가능한 텐서의 값을 <font color="#ffff00">스칼라 값으로 반환</font>합니다. 즉, **텐서가 단 하나의 값을 가지고 있을 때 사용가능** 합니다.

예를 들어
```python
import torch

x = torch.tensor([5])
```

```
5
```

와 같은 텐서는 단 하나의 값을 가지므로 `x.item()` 을 호출하면 5를 반환합니다.

```python
import torch

x = torch.tensor([1,3,5])
```

```
RuntimeError: a Tensor with 3 elements cannot be converted to Scalar
```

또한 위와 같이 여러 값이 들어있는 텐서에서 `x.item()` 을 호출하면 에러가 발생합니다.

또한 <font color="#ffff00">CPU 상에 있는 텐서에 대해서만 사용</font> 할 수 있습니다. GPU 상에 있는 텐서에 대해서는 `.item()` 대신 [[torch.tolist]] 등의 메서드를 사용해야 합니다.

```python
import torch
x = torch.ones(1)
print(x)
print(x.item())
```

```
tensor([1.])
1.0
```

