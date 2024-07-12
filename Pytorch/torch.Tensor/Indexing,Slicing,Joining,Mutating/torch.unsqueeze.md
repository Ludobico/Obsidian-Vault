`.unsqueeze(dim)` 메서드는 텐서의 특정 위치(dim)에 <font color="#ffff00">새로운 차원을 추가하는 메서드</font>입니다.
<font color="#ffff00">새로운 차원의 크기는 1</font>이 됩니다.

> torch.unsqueeze(_input_, _dim_) -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] 

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 인풋 텐서입니다.

> dim -> int
- 추가로 들어갈 차원입니다.


예를 들어, 크기가 (3,4) 인 2차원 텐서 `x` 가 있다고 가정할때 `x.unsqueeze(0)` 을 실행하면, 결과적으로 크기가 (1,3,4) 인 3차원 텐서가 반환됩니다. 이떄 `dim=0` 은 첫번째 차원에 새로운 차원을 추가하라는 뜻입니다.

```python
import torch

x = torch.randn((3))
print(x.size())

print(x.unsqueeze(0).size())
print(x.unsqueeze(0))
```

```
torch.Size([3])
torch.Size([1, 3])
tensor([[-0.4292,  2.4771, -0.3261]])
```

