`torch.nn.Bilinear` 은 [[Pytorch]] 에서 제공하는 또 다른 선형 변환 레이어입니다. 이 레이어는 두 개의 입력 텐서와 가중치 행렬 두 개를 사용하여 <font color="#ffff00">이차 다항식의 특성을 학습하는데 사용</font>됩니다.

`torch.nn.Bilinear` 는 다음과 같은 파라미터를 사용합니다.

> in1_features -> int
- 첫 번째 입력의 특성 차원입니다.

> in2_features -> int
- 두 번째 입력의 특성 차원입니다.

> out_features -> int
- 출력 특성 차원입니다.

> bias -> bool, (optional)
- bias를 사용할지 여부를 나타내는 불리언 값입니다. 기본값은 <font color="#ffc000">True</font> 입니다.

`torch.nn.Bilinear` 은 두 개의 입력 텐서를 받고, 이들 간의 이차 다항식 변환을 수행합니다. 각 입력은 가중치 행렬로 곱해지고, 결과적으로 하나의 출력 텐서가 생성됩니다.

```python
m = nn.Bilinear(20, 30, 40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = m(input1, input2)
print(output.size())
```
```bash
torch.Size([128, 40])
```

