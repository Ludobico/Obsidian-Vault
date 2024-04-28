

.to() 메서드는 텐서를 <font color="#ffff00">지정된 장치(CPU or GPU)로 이동시키는 메서드</font>입니다. 이 메서드는 다음과 같은 방법으로 사용할 수 있습니다.

```python
x = torch.tensor([[1, 2], [3, 4]])
# CPU로 이동
x_cpu = x.to('cpu')
# GPU로 이동
x_gpu = x.to('cuda')
```

또한 데이터 타입을 변환하거나, 장치와 동시에 데이터 타입도 변환할 수 있습니다.

```python
# float 타입으로 변환
x_float = x.to(torch.float32)
# GPU로 이동하면서 float 타입으로 변환
x_gpu_float = x.to('cuda', torch.float32)
```