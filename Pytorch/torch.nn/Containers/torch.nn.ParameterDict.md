`torch.nn.ParameterDict` 는 [[Pytorch]] 에서 사용되는 <font color="#ffff00">모델 파라미터를 딕셔너리 형태로 관리하는 클래스</font>입니다. 이 클래스는 모델 파라미터를 딕셔너리에 저장하고, 각 파라미터는 명확하게 등록되어 모든 모듈 메서드에서 접근 가능합니다. 파라미터들은 모델의 가중치(weight) 및 편향(bias)과 같이 모델의 학습 중에 업데이트 되는 값들을 나타냅니다.

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
                'left': nn.Parameter(torch.randn(5, 10)),
                'right': nn.Parameter(torch.randn(5, 10))
        })

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x
```

