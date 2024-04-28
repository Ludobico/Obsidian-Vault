`.squeeze()` 메서드는 <font color="#ffff00">텐서에서 크기가 1인 차원을 제거</font>합니다.

- 인자가 없으면 모든 크기가 1인 차원을 제거합니다.
- 인자로 차원 인덱스를 지정하면 해당 차원의 크기가 1일 경우 제거합니다.
- squeeze(0) 은 첫 번째 차원이고, squeeze(-1)은 마지막 차원입니다.

```python
import torch

x = torch.randn((3,1,5,4))
print(x.dim())

y = x.squeeze()
print(y.dim())
print(y.size())
```

```
4
3
torch.Size([3, 5, 4])
```

`squeeze(dim)` 에 해당 `dim` 의 크기가 1이 아니여도 에러가 나오지 않고 텐서는 그대로 유지됩니다.

