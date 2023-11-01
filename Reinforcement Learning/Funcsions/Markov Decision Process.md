마르코프 결정 과정(Markov Decision Process, MDP)은 기존 [[Markov Reward Process]] 에서 <font color="#ffff00">행동(Action)이 추가된 확률 모델</font>입니다. MDP 목표는 정의된 문제에 대해 각 상태마다 전체적인 보상을 최대화하는 행동이 무엇인지 결정하는 것입니다. 이때 각각의 상태마다 행동 분포(행동이 선택될 확률)를 표현하는 함수를 <font color="#00b050">정책(policy</font>, $\pi$<font color="#00b050">)</font> 이라고 하며, $\pi$ 는 주어진 상태 $s$ 에 대한 행동 분포를 표현한 것으로 수식은 다음과 같습니다.

$$\pi(a|s) = P(A_t = a | S_t = s)$$
MDP 가 주어진 $\pi$ 를 따를 때 $s$ 에서 $s'$ 로 이동할 확률은 다음 수식으로 계산됩니다.
$$P^{\pi}_{ss'\pi} = \Sigma_{a \in A}\pi(a|s)P^a_{ss'}$$

