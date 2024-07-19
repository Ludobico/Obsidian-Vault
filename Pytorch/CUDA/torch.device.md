`torch.device`  클래스는 [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] 에 **할당되는 장치를 나타내는 객체**입니다. 이 객체는 장치의 유형(일반적으로 `cpu` 또는 `cuda` 이며, `mps`, `xpu`, `xla`, `meta` 도 가능)과 해당 장치 유형의 선택적 장치 번호를 포함합니다. 만약 장치 번호가 없다면, 이 객체는 항상 해당 장치 유형의 현재 장치를 나타냅니다.

`torch.Tensor` 의 장치는 `Tensor.device` 속성을 통해 접근할 수 있습니다.

```python
>>> torch.device('cuda:0')
device(type='cuda', index=0)

>>> torch.device('cpu')
device(type='cpu')

>>> torch.device('mps')
device(type='mps')

>>> torch.device('cuda')  # 현재 CUDA 장치
device(type='cuda')
```

```python
>>> torch.device('cuda', 0)
device(type='cuda', index=0)

>>> torch.device('mps', 0)
device(type='mps', index=0)

>>> torch.device('cpu', 0)
device(type='cpu', index=0)
```

```python
>>> with torch.device('cuda:1'):
...     r = torch.randn(2, 3)
>>> r.device
device(type='cuda', index=1)
```

