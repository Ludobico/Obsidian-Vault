`torch.empty` 함수는 **지정된 크기와 옵션으로 초기화되지 않은 데이터로 채워진 텐서를 생성**하는 [[Pytorch]] 함수입니다. 이 함수는 구조는 정의되지만, 실제 데이터는 초기화되지 않은 상태로 남아 있어 **메모리 상에 이미 존재하는 값**들로 채워집니다. `torch.empty` 연산을 수행하기 전에 텐서에 값을 명시적으로 할당할 계획이 있을 때 유용하게 사용될 수 있습니다.

> torch.empty(_*size_, _*_, _out=None_, _dtype=None_, _layout=torch.strided_, _device=None_, _requires_grad=False_, _pin_memory=False_, _memory_format=torch.contiguous_format_) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")

> size -> int
- 출력 텐서의 형태를 정의하는 정수 시퀀스입니다. 함수 호출 시 가변 인자로 제공되거나 리스트, 튜플 등의 컬렉션으로 제공될 수 있습니다.

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 텐서를 지정할 수 있습니다.이 파라미터가 제공되면, 해당 텐서에 결과가 저장됩니다.

> dtype -> [[torch.dtype]] , optional
- 반환된 텐서의 데이터 타입을 지정할 수 있습니다. 지정하지 않을 경우, `torch.get_default_dtype()` 을 사용합니다.

> layout -> torch.layout, optional
- 텐서의 메모리 레이아웃을 지정합니다. 기본값은 `torch.strided` 로, 일반적인 연속 메모리 레이아웃입니다.

> device -> torch.device, optional
- 텐서를 생성할 디바이스를 지정합니다. 기본적으로 현재 설정된 기본 디바이스에서 생성됩니다.

> requires_grad -> bool, optional, default : False
- 생성된 텐서에 자동 미분(autograd)이 기록을 할지 여부를 설정합니다. 기본값은 `False` 입니다.

> pin_memory -> bool, optional, Default : False
- CPU 텐서에 대해 메모리 핀을 설정할지 여부를 지정합니다. 핀 메모리는 호스트와 디바이스 간의 데이터 전송을 더 효율적으로 만듭니다.

> memory_format -> torch.memory_format, optional
- 반환된 텐서의 메모리 포맷을 지정합니다. 기본값은 `torch.contiguous_format` 입니다.

```python
import torch

torch.cuda.empty_cache()
empty_tensor = torch.empty(3, 4)
print("3x4 Empty Tensor:")
print(empty_tensor)

# 데이터 타입과 디바이스를 지정하여 텐서 생성
empty_tensor_custom = torch.empty(3, 4, dtype=torch.float64, device='cpu')
print("\n3x4 Empty Tensor with custom dtype and device:")
print(empty_tensor_custom)
```

```
3x4 Empty Tensor:
tensor([[9.4498e+21, 1.4097e-42, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

3x4 Empty Tensor with custom dtype and device:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]], dtype=torch.float64)
```

