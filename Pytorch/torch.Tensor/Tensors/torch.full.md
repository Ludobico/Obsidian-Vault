
`torch.full` 함수는 [[Pytorch]] 에서 **특정 크기와 값으로 채워진 텐서를 생성하는 데 사용**됩니다. 이 함수는 다양한 파라미터를 받아 텐서의 크기, 채우는 값, 데이터 타입, 디바이스, 레이아웃 등을 설정할 수 있습니다.

```python
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

> size -> int
- 출력 텐서의 모양을 정의하는 정수들의 리스트, 튜플 또는 [[torch.size]] 객체입니다. 예를들어, (2, 3)은 2x3 텐서를 의미합니다.

> fill_value -> scalar
- 출력 텐서를 채울 값. 이 값은 스칼라여야 합니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 텐서입니다.

> dtype -> [[torch.dtype]] , optional
- 반환된 텐서의 데이터 타입. 기본값은 `None` 이며, 이는 `fill_value` 의 데이터 타입을 사용합니.

> layout -> [[torch.layout]], optional
- 반환된 텐서의 레이아웃입니다. 기본값은 `torch.strided` 입니다.

> device -> 