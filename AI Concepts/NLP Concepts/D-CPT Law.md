- [[#Estimate C|Estimate C]]
- [[#Calculate $D$|Calculate $D$]]


![[Pasted image 20250822144704.png]]

<font color="#ffff00">D-CPT Law(Domain-specific Continual Pre-Training Law)</font> 는 기존의 [[Chinchilla Scaling Law]] 를 확장한 법칙으로, **도메인 특화 추가 학습(DAPT/D-CPT)** 에서 모델 성능을 예측하고 최적의 데이터 혼합 비율을 결정하기 위해 제안되었습니다.

논문에서는 D-CPT Law가 Chinchilla Scaling Law를 기반으로 하여 mixture ratio $r$ 을 도입한 형태라고 설명합니다. Scaling Law는 일반적으로 모델 크기(파라미터 수, $N$), 데이터셋 크기(토큰 수 ,$D$) 그리고 손실(Loss, $L$) 간의 관계를 모델링합니다. D-CPT Law는 mixture ratio $r$ 을 추가하여 일반 코퍼스와 도메인 특화 코퍼스의 혼합 비율에 따른 손실을 예측합니다.

$$
L(N, D, r) = E + \frac{A}{N^{\alpha}} + \frac{B \cdot r^{\eta}}{D^{\beta}} + \frac{C}{(r+\epsilon)^{\gamma}}
$$

## Estimate C

여기서 $C$ 항은 상수값이 아니라, **데이터 분포와 도메인 적합성에 따라 달라지는 보정** 값입니다. 기존의 친칠라 스켈과 달리, D-CPT는 추가 학습 시나이로에서 도메인 데이터의 특수성을 반영해야 하기 때문에 동적으로 추정해야합니다.

$$
C = (L - E - \frac{A}{N^{\alpha}} - \frac{B \cdot \gamma^{\eta}}{D^{\beta}}) \cdot (r + \epsilon)^{\gamma}
$$

## Calculate $D$ 

$$
\begin{aligned}
L_{\text{target}} = E + \frac{A}{N^{\alpha}} + \frac{B \cdot r^{\eta}}{D^{\beta}} + \frac{C}{(r+\epsilon)^{\gamma}} \\\\ D = (\frac{B \cdot r^{\eta}}{L_{\text{target}}-E-\frac{A}{N^{\alpha}}-\frac{C}{(r+\epsilon)^{\gamma}}})^{\frac{1}{\beta}}
\end{aligned}
$$

