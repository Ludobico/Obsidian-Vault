- [[#Main features|Main features]]
- [[#Quantization Format|Quantization Format]]
	- [[#Quantization Format#BF16|BF16]]
	- [[#Quantization Format#Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0|Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0]]
	- [[#Quantization Format#IQ4_NL, IQ4_XS|IQ4_NL, IQ4_XS]]
	- [[#Quantization Format#UD-IQ1_M, UD-Q2_K_XL (Unsloth Dynamic)|UD-IQ1_M, UD-Q2_K_XL (Unsloth Dynamic)]]


GGUF는 [Georgi Gerganov](https://github.com/ggerganov "Georgi Gerganov") 란 개발자가 만든 `ggml` 라이브러리의 후속 포맷입니다. 이는 모델 가중치, 메타데이터, 텐서 정보를 **단일 바이너리 파일에 저장**하여 메모리 효율성과 호환성을 높이는 포맷입니다.

## Main features

- 모델 가중치를 다양한 양자화 포맷(2bit, 4bit, 8bit)으로 저장 가능
- CPU 및 GPU에서 효율적인 인퍼런스를 지원
- `llama.cpp` 와 같은 오픈소스 프로젝트에서 널리 사용됨.

## Quantization Format
![[Pasted image 20250711115950.png]]

파일 이름에 나타나는 양자화 포맷은 **비트 수**와 **양자화 방식의 특성**을 조합한 것입니다. 

### BF16

bfloat16 포맷으로, 16비트 부동소수점 형식을 사용합니다. 이는 양자화가 아닌 원본 모델에 가까운 고정밀 포맷으로, 메모리 사용량이 크지만 정확도가 높습니다.

### Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0

- <font color="#ffff00">숫자(Q2, Q3, Q4)</font> 가중치를 나타내는 비트 수(2bit, 3bit, 3bit)를 의미합니다. 숫자가 작을수록 메모리 사용량이 적지만 정확도가 낮아질 가능성이 높습니다.

- <font color="#ffff00">K</font> : K-means 기반 양자화를 의미합니다. 이는 가중치를 클러스터링하여 효율적으로 양자화하는 방법으로, 정확도가 메모리 효율성을 제공합니다.

- <font color="#ffff00">S, M, L, XL</font> : unsloth 에서 비공식적으로 사용하는 태그입니다. 양자화의 세부 설정을 나타냅니다.
	- S : 더 적은 메모리를 사용하도록 최적화
	- M : 중간 수준의 정확도와 메모리 사용량
	- L : 더 많은 메모리를 사용하지만 정확도가 높음
	- XL : 가장 높은 품질의 양자화 설정

### IQ4_NL, IQ4_XS

 `IQ4_NL` 은 Importance-aware Quantization 방식 중 하나로, 비선형 스케일링을 적용하여, 낮은 비트 수서도 상대적으로 높은 정확도를 유지하려는 목적을 가집니다.

`IQ4_XS` 는 중요도 기반 양자화의 초소형(Extra Small) 프리셋으로, 메모리 사용량을 극단적으로 줄이기 위한 설정입니다.

### UD-IQ1_M, UD-Q2_K_XL (Unsloth Dynamic)

`UD-` 접두사는 Unsloth 프로젝트에서 자체적으로 정의한 Dynamic Quantization 포맷을 의미합니다.

`IQ1`, `IQ2` 는 각각 1bit 및 2bit 양자화 방식으로, 일반적인 LLM 양자화 방식보다 훨씬 낮은 비트수를 사용합니다.

