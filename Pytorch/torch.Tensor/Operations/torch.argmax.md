```python
torch.argmax(_input_) → LongTensor
```

`torch.argmax()` 메서드는 주어진 텐서에서 <font color="#ffff00">최댓값의 인덱스를 반환하는 함수</font>입니다. 

예를 들어, 다음은 간단한 예시입니다.

```python
import torch

input_tensor = torch.randn(5)

print(input_tensor)

max_index = torch.argmax(input_tensor)

print(max_index)
```

```
tensor([-0.2648,  0.4677,  0.1136, -1.1935,  1.7041])
tensor(4)
```

여기서 주어진 tensor 값 중 가장큰 `1.7041` 의 인덱스인 4를 반환하는 것을 확인할 수 있습니다.

여기서 [[torch.item]] 을 사용하여 텐서값을 스칼라 값으로 변환할 수 있습니다.

`torch.argmax()` 메서드에서는 추가적인 옵션을 제공하여 <font color="#ffff00">차원별로 최댓값의 인덱스</font>를 찾을 수 있습니다. 이를 통해 다차원 배열의 경우 원하는 축(axis)을 기준으로 최댓값을 찾을 수 있습니다.

예를 들어, 다차원 배열에서 각 행(axis=1) 또는 각 열(axis=0)에서 최댓값의 인덱스를 찾을 수 있습니다.

```python
import torch

input_tensor = torch.randn(4,3, dtype=torch.float16)

print(input_tensor)

max_indices_rows = torch.argmax(input_tensor, dim=1)

print("각 행에서의 최대 인덱스 : {0}".format(max_indices_rows))

max_indices_cols = torch.argmax(input_tensor, dim=0)

print("각 열에서의 최대 인덱스 : {0}".format(max_indices_cols))
```

```
tensor([[ 0.4578,  1.6855, -1.0430],
        [ 0.2778,  0.3188,  2.9004],
        [-0.8691, -0.3623,  0.2242],
        [-0.0743,  0.5781, -0.1978]], dtype=torch.float16)
각 행에서의 최대 인덱스 : tensor([1, 2, 2, 1])
각 열에서의 최대 인덱스 : tensor([0, 0, 1])
```

