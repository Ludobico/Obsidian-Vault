
`torch.logspace` 는 [[Pytorch]] 에서 제공하는 함수로서, **특정 로그 스케일에 기반하여 일정 범위의 값을 찾는 일차원 텐서를 생성**하는 기능을 제공합니다. 이 함수는 주어진 범위 내에서 로그 스케일 기반으로 균일하게 간격을 둔 값들을 생성하여 그 결과를 텐서 형태로 반환합니다.

주요 파라미터는 다음과 같습니다.

> start -> float or [[Tensors]]
- 로그 스케일의 시작 지점을 나타냅니다. 만약 텐서로 제공될 경우, 0차원이어야합니다.

> end -> float or [[Tensors]]
- 로그 스케일의 끝 지점을 나타냅니다. 이 역시 텐서로 제공될 경우, 0차원이어야합니다.

> steps -> int
- 생성할 텐서의 크기, 즉 값의 개수를 지정합니다.

> base -> float, optioanl
- 로그 스케일의 기준이 되는 밑수입니다. 기본값은 10.0 입니다.

```python
import torch
# 10^0부터 10^2까지 로그 스케일로 50개의 포인트 생성
t = torch.logspace(start=0, end=2, steps=50)
print(t)
```

```
tensor([  1.0000,   1.0985,   1.2068,   1.3257,   1.4563,   1.5999,   1.7575,
          1.9307,   2.1210,   2.3300,   2.5595,   2.8118,   3.0888,   3.3932,
          3.7276,   4.0949,   4.4984,   4.9417,   5.4287,   5.9636,   6.5513,
          7.1969,   7.9060,   8.6851,   9.5410,  10.4811,  11.5140,  12.6486,
         13.8950,  15.2642,  16.7683,  18.4207,  20.2359,  22.2300,  24.4205,
         26.8270,  29.4705,  32.3746,  35.5648,  39.0694,  42.9193,  47.1487,
         51.7947,  56.8987,  62.5055,  68.6649,  75.4312,  82.8643,  91.0298,
        100.0000])
```

