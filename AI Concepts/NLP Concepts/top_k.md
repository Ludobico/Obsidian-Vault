
## top_k(top_k sampling)
---
확률 분포에서 <font color="#ffff00">상위 k개의 다음 단어만 고려하여 선택</font>합니다.

[[temperature]] 에서 다음 텐서가 출력되었다고 가정하면

```
tensor([0.2144, 0.0891, 0.0189, 0.6655, 0.0121])
```

`top_k = 2` 일 경우 다음 2개의 텐서만 출력됩니다.

```
tensor([0.2144, 0.6655])
```

$$
P_{\text{topk}}(x_i) = \frac{P(x_i)}{\Sigma_{x_j \in V_k}P(x_j)}, V_k = \text{arg}(\text{topk}_{x_i \in V})P(x_i)
$$

