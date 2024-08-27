
`Batch Normalization` 은 딥러닝 모델의 학습 속도를 높이고, 훈련 과정에서의 불안정을 줄이며, 더 높은 학습률을 사용할 수 있도록 도와주는 기법입니다. 이는 각 배치에 대해 이벽을 정규화하여 신경망의 층이 받는 **입력의 분포를 일정하게 유지**합니다.

이는 더 높은 학습률을 사용할 수 있게하고, [[Gradient Vanishing]] 문제를 완화하여 깊은 신경망의 학습을 더 안정적으로 만듭니다.

## Input Normalization

- 각 배치에 대해 입력의 <font color="#ffff00">평균</font>과 <font color="#ffff00">분산</font>을 계산합니다.
- 이를 사용하여 각 입려을 정규화하여 평균이 0이고, 분산이 1이 되도록 합니다.

## Trainable Parameters

- 정규화된 입력에 학습 가능한 스케일 파라미터 $\gamma$와 시프트 파라미터 $\beta$를 적용하여 모델의 표현력을 유지합니다.

- 정규화된 출력
$$
\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

- 변환된 출력
$$
y = \gamma\hat{x} + \beta
$$

## Formula of Batch normalization

1. 배치의 평균 계산
$$
\mu B = \frac{1}{m}\Sigma^m_{i=1}x_i
$$

2. 배치의 분산 계산
$$
\sigma^2_B = \frac{1}{m}\Sigma^m_{i=1}(x_i = \mu B)^2
$$

3. 입력 정규화
$$
\hat{x}_i = \frac{x_i - \mu B}{\sqrt{\sigma^2_B+\epsilon}}
$$
4. 스케일 및 시프트 적용
$$
y_i = \gamma\hat{x}_i + \beta
$$


여기서
- $x_i$ 는 배치 내의 입력 데이터입니다.
- $m$은 배치의 크기입니다.
- $\mu B$ 와 $\sigma^2_B$ 는 배치의 평균과 분산입니다.
- $\epsilon$은 분모가 0이 되는 것을 방지하기 위한 작은 값입니다.
- $\gamma$와 $\beta$ 는 학습 가능한 파라미터입니다.

