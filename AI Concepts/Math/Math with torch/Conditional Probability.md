
조건부 확률(Conditional Probability) 은 **어떤 사건이 일어난 후, 다른 사건이 일어날 확률을 구하는 개념**입니다. 예를 들어,

사건 A가 일어난 상황에서 사건 B가 발생할 확률은 다음과 같습니다.

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

여기서

- $P(A|B)$  사건 B가 주어졌을 때 사건 A가 발생할 확률 (조건부 확률)
- $P(A \cap B)$ 사건 A와 사건 B가 동시에 발생할 확률 (교집합)
- $P(B)$ 사건 B가 발생할 확률

## Implementation with pytorch

```python
# 기본적인 조건부 확률 예제 P(A|B) = P(A ∩ B) / P(B)
def basic_conditional_probability():
    P_a = 0.4 # 사건 a
    P_b = 0.5 # 사건 b
    P_A_inter_B = 0.2 # a와 b의 교집합

    P_A_given_B = P_A_inter_B / P_b # 조건부 확률 계산
    print(f"{P_A_given_B:.2f}")

basic_conditional_probability()
```

```
0.4
```

```python
# 샘플 데이터를 활용한 조건부 확률 예제
def sample_based_conditional_probability():
    torch.manual_seed(42)
    data = torch.randint(0, 2, (1000, 2))
    """
    data = 
    tensor([[0, 1],
        [0, 0],
        [0, 1],
        ...,
        [0, 0],
        [1, 0],
        [0, 0]])
    torch.Size([1000, 2])
    """

    # 모든 행, 첫번째 열
    A = data[:, 0]
    # 모든 행, 두번째 열
    B = data[:, 1]

    P_B = (B == 1).float().mean()
    P_A_inter_B = ((A == 1) & (B == 1)).float().mean()
    P_A_given_B = P_A_inter_B / P_B

    print(f"{P_A_given_B:.2f}")


sample_based_conditional_probability()
```

```
0.46
```

