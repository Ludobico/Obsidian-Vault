![[Pasted image 20250721110339.png]]

코사인 유사도는 **두 벡터 간의 방향성을 비교하여 유사도를 측정하는 지표**입니다. 벡터의 크기(길이) 보다는 각도에 초점을 맞추며, 값은 **-1에서 1사이** 입니다.

$$
-1 \le \cos(\theta) \le 1
$$

- 1 : 두 벡터가 동일한 방향 (완벽한 유사)
- 0 : 두 벡터가 직교 (관련성 없음)
- -1 : 두 벡터가 반대 방향

RAG 에서는 쿼리와 문서의 임베딩 벡터를 비교하여, 쿼리에 가장 적합한 문서를 찾기 위해 코사인 유사도를 사용합니다.

## Cosine similarity formula

코사인 유사도는 두 벡터 $A$ 와 $B$ 의 코사인 값을 계산합니다. 수학적으로는 다음과 같이 정의됩니다.

$$
\text{Cosine Similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}
$$

- 분자 : $A \cdot B$ 는 두 벡터 $A$ 와 $B$ 의 **내적** [[dot product]] 입니다.

- 분모 : $\|A\|\|B\|$ 는 각 벡터의 ([[L2, Norm]], 벡터의 크기) 의 곱입니다.

예시로 두 벡터 $A = [1, -2], B = [2,3]$ 을 예로 들면

1. 내적 계산
$$
A \cdot B = -4
$$


2. 크기 계산
$$
\begin{aligned}
\|A\| &= \sqrt{5} \approx 2.236 \\
\|B\| &= \sqrt{13} \approx 3.605
\end{aligned}
$$

3. 코사인 유사도

$$
\text{Cosine Similarity} = \frac{-4}{8.06} \approx -0.4962
$$

내적 값이 음수이므로 코사인 유사도가 음수가 되어, 두 벡터가 반대 방향에 가까움을 나타냅니다.

코사인 유사도 값을 실제 각도로 보고 싶다면 **역삼각함수**를 사용하면 됩니다.

$$
\theta =  \cos^{-1} \times \frac{180}{\pi}
$$

```python
import math

cos_sim = -0.4962
theta_rad = math.acos(cos_sim)  # 역코사인 (라디안)
theta_deg = math.degrees(theta_rad)  # 라디안 → 도(degree) 변환

print(theta_deg)  # 약 119.7도
```

