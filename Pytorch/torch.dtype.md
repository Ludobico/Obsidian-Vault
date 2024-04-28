[[Pytorch]] 에서 `torch.dtype` 은 [[Pytorch/torch.Tensor/torch.Tensor]] 의 **데이터 유형을 정의하는데 사용**됩니다. 파이토치는 다양한 데이터 유형을 지원하며, 이는 모델의 메모리 사용량과 계산 정밀도에 영향을 미칩니다.

`torch.dtype` 에는 다음과 같은 유형들이 있습니다.

---
### Float Types

`torch.float32` `torch.float` : 32비트 부동 소수점

`torch.float64` `torch.double` : 64비트 부동 소수점

`torch.float16` `torch.half` : 16비트 반정밀도(half-precision) 부동 소수점

### Integral Types

`torch.uint8` : 8비트 부호 없는 정수

`torch.int8` : 8비트 부호 있는 정수

`torch.int16` `torch.short` : 16비트 부호 있는 정수

`torch.int32` `torch.int` : 32비트 부호 있는 정수

`torch.int64` `torch.long` : 64비트 부호 있는 정수

### Boolean Types

`torch.bool` : 불리언 값(`True` or `False`)

---

일반적으로 딥러닝 모델에서는 `torch.float32` 와 `torch.float16` 유형이 많이 사용됩니다. `torch.float32` 는 높은 정밀도를 제공하지만 메모리 소비가 크고, `torch.float16` 은 메모리 효율성은 높지만 정밀도가 낮습니다.

텐서의 데이터 유형은 생성 시 `dtype` 인수를 지정하거나, 기존 텐서에서 `tensor.dtype` 속성을 통해 확인할 수 있습니다.

