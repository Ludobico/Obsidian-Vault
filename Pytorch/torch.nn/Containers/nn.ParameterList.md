`torch.nn.ParameterList` 는 [[Pytorch]] 에서 사용되는 <font color="#ffff00">모델 파라미터를 리스트 형태로 관리하는 클래스</font>입니다. 이 클래스는 모델 파라미터를 리스트에 저장하고, 각 파라미터는 명확하게 등록되어 모든 모듈 메서드에서 접근 가능합니다. 파라미터들은 모델의 가중치(weight) 및 편향(bias) 등을 나타내며, 이러한 파라미터는 모델 학습 중에 업데이트되는 값들입니다.

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ParameterList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x
```

