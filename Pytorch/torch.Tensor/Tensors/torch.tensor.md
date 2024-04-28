
`torch.tensor` 는 [[Pytorch]] 에서 **텐서를 생성하는 가장 기본적인 함수 중 하나**입니다. 이 함수를 사용하여 데이터에서 직접 텐서를 생성할 수 있으며, 사용자가 제공한 데이터를 기반으로 새로운 텐서 객체를 만듭니다.

```python
import torch

# 리스트를 사용하여 텐서 생성
data = [1, 2, 3]
tensor_from_list = torch.tensor(data)
print("Tensor from list:", tensor_from_list)

# NumPy 배열을 사용하여 텐서 생성
import numpy as np
np_array = np.array([1, 2, 3])
tensor_from_numpy = torch.tensor(np_array)
print("Tensor from numpy array:", tensor_from_numpy)

# dtype을 지정하여 텐서 생성
tensor_with_dtype = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print("Tensor with dtype:", tensor_with_dtype)

# 다차원 배열
matrix = [[1, 2], [3, 4]]
tensor_matrix = torch.tensor(matrix)
print("Tensor matrix:", tensor_matrix)

```

```
Tensor from list: tensor([1, 2, 3])
Tensor from numpy array: tensor([1, 2, 3], dtype=torch.int32)
Tensor with dtype: tensor([1., 2., 3.])
Tensor matrix: tensor([[1, 2],
        [3, 4]])
```

- 입력 데이터는 균일한 타입을 가져야 합니다. (모두 정수 or 모두 실수)
- `torch.tensor` 는 입력 데이터의 복사본을 만듭니다. 따라서 원본 데이터를 변경해도 텐서에 영향을 주지 않습니다.

