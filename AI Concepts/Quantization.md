- [[#Formula|Formula]]
	- [[#Formula#Integer quantization : Asymmetic-signed|Integer quantization : Asymmetic-signed]]
		- [[#Integer quantization : Asymmetic-signed#Scale & zero-point|Scale & zero-point]]
		- [[#Integer quantization : Asymmetic-signed#encoding (float -> int)|encoding (float -> int)]]
		- [[#Integer quantization : Asymmetic-signed#decoding (int -> float)|decoding (int -> float)]]
	- [[#Formula#Float-to-Float Quantization|Float-to-Float Quantization]]
- [[#Example code|Example code]]



![[Pasted image 20251013135725.png]]

LLM에서 양자화는 **모델의 가중치와 연산을 낮은 비트로 표현**하여 **메모리 사용량과 계산량을 줄이는 기술**입니다. LLM에는 보통 수십억~수천억 개의 파라미터를 가지고 있어, flaot32 같은 일반 부동 소수점으로 저장하면 메모리와 연산 비용이 매우 큽니다. 양자화를 통해 이러한 데이터를 int8, int4 같은 낮은 비트 정수로 변환하면, 모델 크기를 크게 줄일 수 있을 뿐만 아니라, GPU나 TPU 에서 연산 속도를 높이는 효과도 있습니다.

## Formula
---
![[Pasted image 20251013140204.png]]

양자화는 모델의 <font color="#ffff00">가중치(weight)</font>나 <font color="#ffff00">활성화 값(activation)</font> 같은 고정밀 부동소수점(float32) 데이터를 저정밀 정수(int8, int4)나 부동소수점(float16, bfloat16)으로 변환하는 기술입니다. 핵심 목표는 **메모리 사용량을 줄이고 계산 속도를 높이는 것**이며, 이 과정에서 모델의 성능 저하를 최소화해야 합니다.

이 변환 과정은 미리 계산된 <font color="#ffff00">스케일(scale)</font>과 <font color="#ffff00">영점(zero-point)</font> 이라는 두 가지 핵심 파라미터를 사용하여 실수와 정수 사이의 관계를 정의합니다.

### Integer quantization : Asymmetic-signed

가장 널리 사용되는 `int8` 양자화 표준입니다. 실수 0.0이 정수 0에 매핑되지 않아도 되는 비대칭(asymmetic) 구조를 가지며, 부호 있는(signed) 정수를 사용합니다.

- $w$ : 원본 실수 값
- $q$ : 양자화된 정수 값
- $s$ : 스케일(float)
- $z$ : 영점(정수)
- \[$q_{\text{min}}$, $q_{\text{max}}$\] : 양자화된 정수 데이터 범위 (int8의 경우 \[-128, 127\])

#### Scale & zero-point

인코딩과 디코딩에 사용될 스케일($s$) 와 영점($z$)을 원본 가중치 텐서로부터 미리 계산합니다. 이 두 값은 실수 범위와 정수 범위 사이의 다리 역할을 합니다.

- 스케일(Scale, $s$)
실수 범위의 포을 정수 범위의 폭으로 나눈 값으로, 정수 1칸이 실수 얼마만큼의 크기를 갖는지를 나타냅니다.
$$
s = \frac{w_{\text{max}} - w_{\text{min}}}{q_{\text{max}} - q_{\text{min}}}
$$

- 영점(zero-point, $z$)
실수 0.0이 어떤 정수값에 매핑되는지를 나타냅니다.

$$
z = \text{round}(q_{\text{min}} - \frac{w_{\text{min}}}{s})
$$
#### encoding (float -> int)

$$
q = \text{round}(\frac{w}{s} + z)
$$

실수를 스케일로 나눈 뒤 영점을 더해 정수 범위로 옮기고, 가장 가까운 정수로 반올림 합니다.

#### decoding (int -> float)

$$
w^{'} = s \times (q - z)
$$
정수에서 영점을 빼고 스케일을 ㅗㅂ하여 원래의 실수 범위로 되돌립니다. 여기에서 나온 값 $w^{'}$ 은 양자화 오차로 인해 원본 $w$ 와 미세한 차이를 가집니다.

### Float-to-Float Quantization

`float32` 를 `float16` 이나 `bfloat16` 으로 변환하는 과정입니다. 이는 선형 변환이 아닌, 실수의 비트 구조를 재할당하는 방식으로 이루어집니다.

| 데이터 타입 | 부호(Sign) | 지수(Exponent) | 가수(Mantissa) | 총 비트    | 특징                          |
| ------ | -------- | ------------ | ------------ | ------- | --------------------------- |
| FP32   | 1 bit    | 8 bits       | 23 bits      | 32 bits | 표준 단정밀도                     |
| FP16   | 1 bit    | 5 bits       | 10 bits      | 16 bits | 정밀도 높음, 표현 범위 좁음            |
| BF16   | 1 bit    | 8 bits       | 7 bits       | 16 bits | 표현 범위 넓음 (FP32와 동일), 정밀도 낮음 |

**LLM에서의 트렌드:** LLM 학습 및 추론 시에는 **넓은 표현 범위가 중요**하기 때문에, 정밀도를 약간 희생하더라도 FP32의 범위를 그대로 유지하는 **`bfloat16`이 사실상의 표준**으로 사용되고 있습니다.

## Example code

```python
import torch

torch.manual_seed(42)

# randn으로 float32 텐서 생성
# 평균 0에 표준편차 2.5의 정규분포 텐서
original_tensor = torch.randn(10) * 2.5
print(f"✅ 1. 원본 텐서 (float32) : {original_tensor.numpy()}")

# int 8 데이터 타입의 범위
q_min, q_max = -128, 127

# 텐서의 최소, 최댓값
w_min, w_max = original_tensor.min(), original_tensor.max()

# 스케일(s) 계산
scale = (w_max - w_min) / (q_max - q_min)

# 영점(z) 계산
zero_point_float = q_min - (w_min / scale)
zero_point = int(torch.round(zero_point_float).item())
print(f"✅ 2. 양자화 파라미터:")
print(f"   - 스케일 (s): {scale.item():.6f}")
print(f"   - 영점 (z): {zero_point}\n")

# 3. 양자화 (Encoding): float32 -> int8
quantized_tensor = torch.round(original_tensor / scale + zero_point)
quantized_tensor = torch.clamp(quantized_tensor, q_min, q_max)
quantized_tensor = quantized_tensor.to(torch.int8)

print(f"✅ 3. 양자화된 텐서 (int8):\n{quantized_tensor.numpy()}\n")

# 4. 역양자화 (Decoding): int8 -> float32
dequantized_tensor = scale * (quantized_tensor.to(torch.float32) - zero_point)

print(f"✅ 4. 역양자화된 텐서 (float32):\n{dequantized_tensor.numpy()}\n")

# 5. 결과 비교
quantization_error = torch.abs(original_tensor - dequantized_tensor)
print(f"✅ 5. 양자화 오차 (절대값):\n{quantization_error.numpy()}\n")
```

```
✅ 1. 원본 텐서 (float32) : [ 0.84172595  0.3220235   0.5861559   0.5758326  -2.8071408  -0.46582073
  5.5205035  -1.5949926   1.1541431   0.6683772 ]
✅ 2. 양자화 파라미터:
   - 스케일 (s): 0.032657
   - 영점 (z): -42

✅ 3. 양자화된 텐서 (int8):
[ -16  -32  -24  -24 -128  -56  127  -91   -7  -22]

✅ 4. 역양자화된 텐서 (float32):
[ 0.8490932   0.3265743   0.58783376  0.58783376 -2.808539   -0.457204
  5.5191054  -1.600214    1.14301     0.6531486 ]

✅ 5. 양자화 오차 (절대값):
[0.00736725 0.00455078 0.00167787 0.01200116 0.00139809 0.00861672
 0.00139809 0.00522137 0.01113307 0.01522863]
```

