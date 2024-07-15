`torch.nn.ModuleDict` 는 [[Pytorch]] 에서 사용되는 <font color="#ffff00">모듈을 dictionary 형태로 관리하는 클래스</font>입니다. 이 클래스는 모듈을 key-value 쌍의 형태로 저장하며, 각 모듈은 이 딕셔너리에 등록되어 모든 모듈 메서드에서 접근 가능합니다.

`ModuleDict` 는 여러 모듈을 이름으로 구분하여 관리하거나 여러 모듈을 그룹화하고 조직화하는데 유용합니다. 모델 내의 서로 다른 부분을 이름별로 구분하거나 모듈의 이름을 통해 모듈에 액세스하는 데 활용할 수 있습니다.

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
```

