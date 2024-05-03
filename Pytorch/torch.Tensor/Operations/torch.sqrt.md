`torch.sqrt` 함수는 [[Pytorch]] 텐서의 **각 요소에 대해 제곱근 연산을 수행하는 함수**입니다. 이 함수는 주어진 입력 텐서의 각 요소에 대해 제곱근을 계산하고, 그 결과를 새로운 텐서로 반환합니다.

수식으로 표현하면 다음과 같습니다.

$$out_i = \sqrt{input_i}$$

> torch.sqrt(_input_, _*_, _out=None_) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")

이 함수는 다음과 같은 파라미터를 가집니다.

> input -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]
- 제곱근 연산을 수행하는 입력 텐서

> out -> [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]] , optional
- 결과를 저장할 출력 텐서를 지정합니다. 지정하지 않으면 새로운 텐서가 생성됩니다.

`torch.sqrt` 는 다양한 분야에서 사용됩니다. 예를 들어, 머신러닝 분야에서는 손실 함수 계산, 정규화 등의 과정에서 제곱근 연산이 필요할 수 있습니다. 또한, 수치 해석이나 신호 처리 분야에서도 제곱근 연산이 자주 사용됩니다.

## example code
---

```python
import torch

a = torch.randn(4)
print(a)

b = torch.sqrt(a)
print(b)
```

```
tensor([ 0.2760, -1.0791,  0.7937,  1.4040])
tensor([0.5254,    nan, 0.8909, 1.1849])
```

결과를 보면 입력 텐서의 음수 값에서는 `nan` 이 출력됨을 알 수 있습니다.

음수에 대해 제곱근 연산을 수행하면 nan이 출력되는 이유는 **실수 집합에서 음수의 제곱근이 정의되지 않기 때문**입니다.

실수 집합에서 제곱근 연산은 다음과 같이 정의됩니다.

$$\sqrt{x} = y <=> y^2 = x $$
$$x \geq 0$$

즉, $x$ 가 0 이상의 값일 때만 실수 $y$ 에 대해 $y^2 = x$ 를 만족하는 $y$ 가 존재합니다. 예를 들어 $x=4$ 라면 $y = \pm2$ 가 됩니다.

하지만 $x$ 가 음수라면, 어떤 실수 $y$에 대해서도 $y^2 = x$ 를 만족하는 $y$ 가 존재할 수 없기 때문에, 컴퓨터에서 음수의 제곱근을 계산하려 할때 nan 값을 출력하게 됩니다.

음수의 제곱근을 계산해야 하는 상황이라면, 복소수 영역으로 확장하여 계산해야 합니다. 복소수 영역에서는 음수의 제곱근도 정의될 수 있습니다.

