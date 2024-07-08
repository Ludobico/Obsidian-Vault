`torch.exp` 는 [[Pytorch]] 에서 제공하는 함수로, 주어진 입력 텐서의 각 요소에 대해 지수 함수 $e^x$ 를 계산합니다. 여기서 $e$ 는 자연 로그의 및(약 2.71828)입니다.

$$
e = \lim_{n\rightarrow\infty}(1 + \frac{1}{n})^n
$$

```python
torch.exp(input, *, out=None) -> Tensor
```

> input
- 지수 함수를 적용할 입력 텐서

> out -> optional
- 결과를 저장할 텐서

