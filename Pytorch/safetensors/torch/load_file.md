`safetensors.torch.load_file()` 은 **Safetensors 포맷** 으로 저장된 데이터를 읽어와 **[[Pytorch]] 텐서 형식으로 로드하는 함수** 입니다.

## Parameters

> filename -> str or os.PathLike

Safetensors 파일의 경로를 지정합니다.

> device -> [[torch.device]] , default : 'cpu'

로드된 텐서가 위치할 장치를 지정합니다.
`cpu` 또는 `cuda`, `cuda:0`, `cuda:1` 등을 지정할 수 있습니다.
GPU를 사용하려면 [[Pytorch]] 에서 [[Pytorch/CUDA/CUDA|CUDA]] 를 활성화할 수 있어야 합니다.

---

[[torch.load]] 와는 다르게 `pickle` 을 사용하지 않아 악성 코드 실행 위험이 없습니다.

```python
import safetensors.torch
import torch

# 텐서 생성
data = {"tensor1": torch.randn(3, 3), "tensor2": torch.ones(5)}

# Safetensors 파일로 저장
safetensors.torch.save_file(data, "data.safetensors")
print("Saved as 'data.safetensors'")
```

```python
import safetensors.torch

# Safetensors 파일 로드
loaded_data = safetensors.torch.load_file("data.safetensors", device="cpu")
print("Loaded Tensors:", loaded_data)

# 텐서 출력
for name, tensor in loaded_data.items():
    print(f"{name}: {tensor}")
```

