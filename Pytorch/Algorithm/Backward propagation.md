![[1_VF9xl3cZr2_qyoLfDJajZw.gif]]

Backward Propagation(역전파)은 순전파 후에 가중치 및 편향 등의 <font color="#ffff00">모델의 매개변수를 업데이트하기 위해 사용되는 과정</font>입니다.

순전파에서는 입력 데이터가 모데을 통과하면서 예측값이 계산되고, 손실 함수를 통해 예측값과 실제값의 오차를 계산합니다. 그런 다음, 역전파를 통해 이 오차를 이용하여 각 매개변수의 기울기(gradient)를 계산합니다.

역전파는 다음과 같은 단계로 이루어집니다.

1. <font color="#ffc000">손실 함수의 미분</font>
손실 함수를 입력으로 받아, 이를 각 매개변수로 미분합니다. 이를 통해 손실 함수의 값을 각 매개변수에 대한 기울기로 반환합니다.

2. <font color="#ffc000">기울기 계산</font>
역전파 단계에서는 체인룰(chain rule)을 사용하여 출력층에서 입력층으로 거슬러 올라가며, 각 층의 기울기를 계산합니다. 각 층에서는 이전 층으로부터 전달된 기울기와 해당 층의 활성화 함수의 미분 값을 곱하여 기울기를 계산합니다.

3. <font color="#ffc000">매개변수 업데이트</font>
계산된 기울기를 사용하여 모델의 매개변수를 업데이트합니다. 일반적으로 경사하강법 또는 그 변종 알고리즘을 사용하여 매개변수를 조정합니다.

4. <font color="#ffc000">반복</font>
순전파와 역전파 단계를 반복하여 모델의 매개변수를 조금씩 조정하고, 손실 함수를 최소화하도록 학습합니다.

따라서 <font color="#ffff00">역전파는 순전파에서 계산된 오차를 역방향으로 전파하고, 이를 통해 각 매개변수의 기울기를 계산하여 모델의 매개변수를 업데이트하는 과정</font>입니다. 이후 다시 순전파를 진행하여 업데이트된 매개변수를 기반으로 새로운 예측값을 계산하고, 이를 반복하여 모델을 학습시킵니다.
