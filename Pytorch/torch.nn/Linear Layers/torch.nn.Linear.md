`torch.nn.Linear` 모듈은 인공 신경망에서 사용되는 <font color="#ffff00">선형 변환(Linear Transformation)을 수행하는 모듈</font>입니다. 이 모듈은 입력 데이터를 선형 방정식을 통해 변환하고 편향(bias)를 더하는 역할을 합니다.

$$y = xA^T + b$$

> in_features -> int
- 입력 데이터의 특성 수 또는 차원 수를 나타내는 정수입니다.

> out_features -> int
- 출력 데이터의 특성 수 또는 차원 수를 나타내는 정수입니다.

> bias -> bool
- 편향을 사용할지 여부를 나타내는 불리언 값입니다. <font color="#ffc000">True</font> 로 설정하면 편향이 사용되고 <font color="#ffc000">False</font>로 설정하면 편향이 사용되지 않습니다.

```python
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
```

```bash
torch.Size([128, 30])
```

