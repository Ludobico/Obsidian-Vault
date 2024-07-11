```python
torch.tranpose(input, dim0, dim1) -> torch.Tensor
```

`torch.transpose` 함수는 [[Pytorch]]의 입력 텐서의 특정 차원들을 교환하여 새로운 텐서를 생성하는 역할을 합니다 이 함수는 <font color="#ffff00">텐서의 차원을 재배열하는 데 사용</font>됩니다.

`torch.transpose` 함수의 파라미터는 다음과 같습니다.

> input -> torch.Tensor
- 함수가 적용될 입력 텐서입니다. 텐서의 모양(shape)를 변경하고 차원을 교환할때 사용됩니다.

> dim0 -> int
- 첫 번째로 교환할 차원의 인덱스입니다. 이 값은 0부터 시작하며, 입력 텐서의 차원중 하나를 선택하여 교환할 때 사용됩니다.

> dim1 -> int
- 두 번째로 교환할 차원의 인덱스입니다. 역시 0부터 시작하는 인덱스이며, 입력 텐서의 다른 차원과 교환할 때 사용됩니다.

```python
import torch


x = torch.randn(2, 3)
print(x)

tx = torch.transpose(x, 0, 1)
print(tx)

```

```
tensor([[-0.4380,  0.5224,  1.4152],
        [-0.1751,  0.9744, -0.1268]])
```

```
tensor([[-0.4380, -0.1751],
        [ 0.5224,  0.9744],
        [ 1.4152, -0.1268]])
```


```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.transpose(0, 1)
print(y)
```

```
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

