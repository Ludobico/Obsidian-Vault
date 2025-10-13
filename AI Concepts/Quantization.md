- [[#Formula|Formula]]
	- [[#Formula#Int Encoding|Int Encoding]]
	- [[#Formula#Int Decoding|Int Decoding]]
	- [[#Formula#float|float]]
- [[#Example|Example]]


![[Pasted image 20251013135725.png]]

LLM에서 양자화는 **모델의 가중치와 연산을 낮은 비트로 표현**하여 **메모리 사용량과 계산량을 줄이는 기술**입니다. LLM에는 보통 수십억~수천억 개의 파라미터를 가지고 있어, flaot32 같은 일반 부동 소수점으로 저장하면 메모리와 연산 비용이 매우 큽니다. 양자화를 통해 이러한 데이터를 int8, int4 같은 낮은 비트 정수로 변환하면, 모델 크기를 크게 줄일 수 있을 뿐만 아니라, GPU나 TPU 에서 연산 속도를 높이는 효과도 있습니다.

## Formula
---
![[Pasted image 20251013140204.png]]

### Int Encoding
실수값 $w$ 를 비트 수 $b$ 의 정수 $q$로 매핑하는 과정

$$
q = \text{round}(\frac{w-w_{\text{min}}}{w_{\text{max}} -w_{\text{min}}} \times (2^b - 1))
$$

- $w_{\text{min}}$ , $w_{\text{max}}$ = 양자화할 값의 최소/최댓값
- $b$ = 비트 수 (예 : int8 = 8, int4 = 4)
- round : 가장 가까운 정수로 반올림

### Int Decoding
양자화된 정수 $q$ 를 다시 실수 근사값 $\hat{w}$ 로 복원하는 과정
$$
\hat{w} = q \times \frac{w_{\text{max}} - w_{\text{min}}}{2^b - 1} + w_{\text{min}}
$$

### float

- encode/decode 수식 필요 없음
- exponent/mantissa 비트 감소로 인한 정밀도 감소, 범위는 거의 그대로 유지

양자화는 단순히 소수점을 버리거나 반올림하는 것이 아니라, **모델의 weight나 activation value을 낮은 비트로 압축하고, 계산 시 다시 근사값으로 복원하는 과정**입니다. 이 과정은 크게 두 단계로 이루어집니다.

먼저, 실수값이 속한 범위를 기준으로 <font color="#ffff00">스케일링</font>을 수행합니다. 모델의 가중치가 특정 범위에 분포한다고 가정하면, 이 범위를 낮은 비트 정수가 표현할 수 있는 단계 수로 나누어 정규화합니다. 예를 들어, int8 양자화라면 -128 부터 127 까지 256단계로 범위을 나누는 것이고, int4라면 16단계로 나누는 식입니다.

## Example
---

임의 가중치
$$
w = 12.3456(\text{float32})
$$

