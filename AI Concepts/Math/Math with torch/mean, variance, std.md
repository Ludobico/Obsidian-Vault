- [[#평균 (Mean)|평균 (Mean)]]
- [[#분산 (Variance)|분산 (Variance)]]
- [[#모집단 표준편차(Population standard Deviation)|모집단 표준편차(Population standard Deviation)]]
- [[#표본 표준편차(Sample standard deviation)|표본 표준편차(Sample standard deviation)]]

## 평균 (Mean)

평균은 모든 값을 더한 후 데이터 개수 $N$ 으로 나누어 계산합니다.

기호는 $\mu$(mu) 또는 $\bar{x}$ 로 표시합니다. 예를 들어 텐서

$$
x = [1,2,3]
$$
이 주어졌을때 평균은

$$
\mu = \frac{1+2+3}{3} = 2
$$

입니다.

```python
tensor1 = torch.tensor([1,2,3])
def torch_mean(tensor : torch.Tensor):
    mean_value = tensor.mean()
    print(f"mean value : {mean_value.item():.3f}")
```


## 분산 (Variance)

분산은 각 데이터가 평균으로부터 얼마나 떨어져 있는지를 **제곱한 값들의 평균**입니다. 표준편차는 분산의 제곱근이므로, 분산은 표준편차의 제곱이라고 할 수 있습니다.

기호는 $\text{var}$ 로 표시합니다.


$$
x = [1,2,3]
$$
$$
\text{var} = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{1+0+1}{3} = \frac{2}{3}
$$

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
sample_variance = tensor1.var(unbiased=True)
```

## 모집단 표준편차(Population standard Deviation)

모든 데이터가 있는 모집단 표준편차는 $N$ 으로 나누어 계산하거나, 분산에 제곱근을 취합니다.

기호는 $\sigma$ 로 표시합니다.

$$
x = [1,2,3]
$$

$$
\sigma = \sqrt{\frac{\Sigma(x_i - \mu)^2}{N}}
$$

$$
= \sqrt{\frac{1+0+1}{3}} = \sqrt{\frac{2}{3}} \approx 0.816
$$

```python
import torch
tensor1 = torch.tensor([1,2,3])
std_value = tensor1.std(unbiased=False)
```

모집단의 분산을 구할 때는 모든 데이터를 다 알고 있기 때문에 단순히 평균으로부터 거리(분산)을 그대로 평균 내면 됩니다.

## 표본 표준편차(Sample standard deviation)

하지만 현실에서 모집단의 모든 데이터를 아는 경우는 거의 없습니다. 그래서 일부 데이터만 표본(sample) 으로 뽑아 표준편차를 추정합니다.

이때는 $N - 1$ 로 나누는 **베셀의 보정(Bessels's Correction)** 을적용하여 표준편차를 구합니다.

기호는 $s$ 로 표시합니다.

$$
x = [1,2,3]
$$

$$
\sigma = \sqrt{\frac{\Sigma(x_i - \mu)^2}{N - 1}}
$$

$$
= \sqrt{\frac{1+0+1}{3}} = \sqrt{\frac{2}{2}} = 1
$$

```python
import torch
tensor1 = torch.tensor([1,2,3])
std_value = tensor1.std(unbiased=True)
```

