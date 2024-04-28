`torch.ones` 함수는 **모든 원소가 1인 텐서를 생성**하는 [[Pytorch]] 함수입니다. 이 함수는 주어진 크기와 옵션에 따라 특정 데이터 타입, 장치, 레이아웃을 갖는 텐서를 생성할 수 있습니다. `torch.ones` 는 초기값 설정, 특정 연산을 위한 텐서 준비 등 다양한 목적으로 사용될 수 있습니다.

> torch.ones(_*size_, _*_, _out=None_, _dtype=None_, _layout=torch.strided_, _device=None_, _requires_grad=False_)

> size -> int
- 출력 텐서의 형태를 정의하는 정수 시퀀스입니다. 함수 호출 시 가변 인자로 제공되거나 리스트, 튜플 등의 컬렉션으로 제공될 수 있습니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 텐서를 지정할 수 있습니다. 이 파라미터가 제공되면, 해당 텐서의 결과가 저장되, 메모리 할당을 최적화할 수 있습니다.

> dtype -> [[torch.dtype]] , optional
- 반환된 텐서의 데이터 타입을 지정할 수 있습니다. 지정하지 않을 경우, `torch.get_default_dtype()` 을 사용합니다.

> layout -> torch.layout, optional
- 텐서의 메모리 레이아웃을 지정합니다. 기본값은 `torch.strided` 로, 일반적인 연속 메모리 레이아웃입니다.

> device -> torch.device, optional
- 텐서를 생성할 디바이스를 지정합니다. 기본적으로 현재 설정된 기본 디바이스에서 생성됩니다.

> requires_grad -> bool, optional, default : False
- 생성된 텐서에 자동 미분(autograd)이 기록을 할지 여부를 설정합니다. 기본값은 `False` 입니다.

```python
import torch

# 3x3 크기의 텐서 생성, 모든 원소가 1
ones_tensor = torch.ones(3, 3)
print("3x3 Ones Tensor:")
print(ones_tensor)

# 다른 데이터 타입을 지정하여 텐서 생성
ones_tensor_int = torch.ones(3, 3, dtype=torch.int32)
print("\n3x3 Ones Tensor with dtype int32:")
print(ones_tensor_int)

```

```
3x3 Ones Tensor:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])

3x3 Ones Tensor with dtype int32:
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
```

