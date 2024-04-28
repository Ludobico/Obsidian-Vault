
`torch.zeros` 함수는 **모든 원소가 0인 텐서를 생성**하는 [[Pytorch]] 함수입니다. 이 함수는 특정 크기와 데이터 타입을 지정하여 호출할 수 있으며, 모델의 가중치 초기화나 다양한 연산에서 임시 버퍼로 사용될 때 유용합니다.

> torch.zeros(_*size_, _*_, _out=None_, _dtype=None_, _layout=torch.strided_, _device=None_, _requires_grad=False_)

```python
import torch

# 3x3 크기의 텐서를 생성, 모든 원소가 0
zero_tensor = torch.zeros(3, 3)
print("3x3 Zero Tensor:")
print(zero_tensor)

# 데이터 타입을 지정하여 텐서 생성
zero_tensor_float64 = torch.zeros(3, 3, dtype=torch.float64)
print("\n3x3 Zero Tensor with dtype float64:")
print(zero_tensor_float64)
```

```
3x3 Zero Tensor:
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])

3x3 Zero Tensor with dtype float64:
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
```

