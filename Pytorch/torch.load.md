
`torch.load()` 는 [[Pytorch]] 에서 **저장된 데이터를 파일로부터 로드하는 함수**입니다. 이 함수는 주로 `torch.save()` 로 저장된 데이터를 다시 읽어오기 위해 사용됩니다. 저장된 데이터는 일반적으로 모델 weights, tensors, 또는 기타 [[Python]] 객체 일 수 있습니다.

## Parameters

> f 

읽어올 파일의 경로(str, os.PathLike) 또는 파일 객체입니다.
파일은 `torch.save()` 로 저장된 형식이어야 합니다.

> map_location

- 저장된 텐서의 장치를 다른 장치로 매핑할 때 사용됩니다.

1. [[torch.device]] 또는 `str`

```python
map_location=torch.device('cpu')  # 모든 텐서를 CPU로 로드
```

2. `dict` : 특정 장치를 다른 장치로 매핑합니다.

```python
map_location={'cuda:0': 'cuda:1'}  # GPU 0에서 저장된 데이터를 GPU 1로 로드
```

> pickle_module

데이터를 로드할 때 사용할  직렬화/역직렬화 모듈입니다.
기본값은 Python의 내장 `pickle` 모듈입니다.
만약 데이터가 커스텀 직렬화 방식으로 저장된 경우, 해당 모듈을 지정해야 합니다.

> weights_only

`pytorch 2.0` 이상에서 지원되며, **[[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]], 기본 타입(dict, list 등) 만 로드** 되로록 제한합니다.
보안과 성능 최적화를 위해 사용됩니다.

> mmap

텐서 데이터가 파일 전체를 메모리에 적재하지 않고 메모리 맵(mmap)을 통해 읽히도록 설정합니다.
메모리 사용량을 줄이고 대용량 파일에서 성능을 향상시킬 수 있습니다.

> pickle_load_args

`pickle.load()` 나 `pickle.Unipickler` 에 전달할 추가 인자를 지정합니다.


## example code

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
print("Original Tensor : ", tensor)

# 간단한 선형 변환
W = torch.tensor([2.0, 0.5, -1.0])
b = torch.tensor(1.0)

transformed_tensor = tensor * W + b
print("Transformed Tensor:", transformed_tensor)

# 텐서 저장
torch.save(transformed_tensor, 'transformed_tensor.pth')

loaded_tensor = torch.load('transformed_tensor.pth', map_location='cpu')
print("Loaded Tensor:", loaded_tensor)
```

```
Original Tensor :  tensor([1., 2., 3.])
Transformed Tensor: tensor([ 3.,  2., -2.])
Loaded Tensor: tensor([ 3.,  2., -2.])
```

![[Pasted image 20250124171819.png]]

