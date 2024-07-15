`torch.nn.ModuleList` 는 [[Pytorch]] 에서 사용되는 모듈의 리스트를 관리하는 클래스입니다. 이 클래스는 <font color="#ffff00">모듈을 리스트에 추가하고 인덱싱 및 반복 가능한 리스트연산을 지원하여 모듈 관리를 단순화</font>합니다.

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

```python
import torch.nn as nn

# ModuleList에 선형 레이어 추가
module_list = nn.ModuleList([
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
])

# ModuleList의 모듈에 인덱싱
linear_layer = module_list[0]
print(linear_layer)

# ModuleList를 반복하여 모든 모듈에 접근
for module in module_list:
    print(module)
```

