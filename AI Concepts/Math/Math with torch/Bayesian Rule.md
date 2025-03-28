
베이즈 룰(Baysian Rule)은 **조건부 확률을 계산하는 방법**으로, 주어진 정보나 데이터를 바탕으로 어떤 사건에 대한 확률을 갱신하는 데 사용됩니다. 기본적으로 베이즈 룰은 다음과 같은 수식으로 표현됩니다.

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

여기서 각 수식은

$$
P(A|B)
$$

- 사건 B가 주어졌을 때 사건 A가 발생할 확률 (후험도, posterior probability)

$$
P(B|A)
$$
- 사건 A가 주어졌을 때 사건 B가 발생할 확률 (우도, likelihood)

$$
P(A)
$$
- 사건 A가 발생할 확률

$$
P(B)
$$
- 사건 B가 발생할 확률

베이즈 룰은 우리가 알고 있는 데이터를 바탕으로 사건의 확률을 갱신하는 데 유용합니다.

