딥 큐-러닝(Deep Q-Learning)은 합성곱 신경망을 이용하여 큐-함수를 학습하는 [[Reinforcement Learning]] 기법입니다. 이떄 합성곱층을 깊게 하여 훈련할 때, <font color="#ffff00">큐 값의 정확도를 높이는 것을 목표</font>로 합니다.

![[Pasted image 20231101170302.png]]

딥 큐-러닝의 특징들을 하나씩살펴보겠습니다.

- 강화 학습을 위한 시뮬레이션 환경을 제공

강화 학습을 위한 시뮬레이션 환경을 구현하는 데 중요한 함수가 세 개 있습니다.
- <font color="#ffc000">reset()</font>
	- 환경을 초기화할 때 사용합니다. 에이전트가 게임을 시작하거나 초기화가 필요할 때 reset() 함수를 사용하며, 초기화될 때는 관찰 변수(상태를 관찰하고 그 정보를 저장)를 함께 반환합니다.

- <font color="#ffc000">step()</font>
	- 에이전트에 명령을 내리는 함수입니다. 따라서 가장 많이 호출되는 함수로, 이 함수로 행동 명령을 보내고 환경에서 관찰 변수, 보상 및 게임 종료 여부 등 변수를 반환합니다.

- <font color="#ffc000">render()</font>
	- 화면에 상태를 표시하는 역할을 합니다.


## <font color="#ffc000">Target Q-Network</font>
---
[[Q-learning]] 에서는 큐-함수가 학습되면서 큐 값이 계속 바뀌는 문제가 있었는데, 딥 큐-러닝에서는 이 문제를 해결하기 위해 <font color="#00b050">타깃 큐-네트워크(target Q-Network)</font> 를 사용합니다. 즉, 큐-네트워크 외에 별도로 타깃 큐-네트워크를 두는 것이 특징입니다. 두 네트워크는 가중치 파라미터만 다르고 완전히 같습니다. DQN에서는 수렴을 원활하게 시키기 위해 타깃 큐-네트워크를 계속 업데이트하는 것이 아니라 주기적으로 한 번씩 업데이트합니다.

![[Pasted image 20231101170758.png]]

훈련을 수행할 떄의 손실 함수로는 MSE를 사용합니다. 네트워크 두 개가 분리되어 있으므로 각 네트워크에서 사용되는 파라미터 $\theta$ 의 표기가 다른 것을 확인할 수 있습니다.

![[Pasted image 20231101170915.png]]
