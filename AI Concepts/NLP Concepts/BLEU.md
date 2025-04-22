
- [[#Core Concept|Core Concept]]
- [[#Formula|Formula]]
		- [[#n-gram 정밀도|n-gram 정밀도]]
- [[#Examples|Examples]]
	- [[#Examples#Bleu-1 (1-gram)|Bleu-1 (1-gram)]]
	- [[#Examples#Bleu-2 (2-gram)|Bleu-2 (2-gram)]]


BLEU(Billingual Evaluation Understudy)는 **기계 번역 및  텍스트 생성 모델의 품질을 평가하기 위한 n-gram 기반 지표**입니다. BLEU는 모델이 생성한 텍스트와 참조 텍스트 간의 <font color="#ffff00">정밀도(precision)</font>를 중심으로 유사성을 측정하며, 주로 기계 번역에서 사용됩니다. [[ROUGE]] 가 <font color="#ffff00">리콜(Recall)</font> 중심인 것과 달리, BLEU는 정밀도(precision)에 초점을 맞춥니다.

## Core Concept

1. N-gram 정밀도(Precision) : 생성된 텍스트에서 참조 텍스트와 일치하는 n-gram의 비율을 측정합니다.
2. Brevity Penalty(BP) : 생성된 텍스트가 참조 텍스트보다 너무 짧을 경우 점수를 낮추는 페널티를 적용합니다.
     - 참조 문장 (Reference)
	 -  "The quick brown fox jumps over the lazy dog"
	 - 생성 문장 (Candidate 1)
	 - "The quick brown fox"
	 - 생성 문장 (Candidate 2)
	 - "The quick brown dog jumps over the cat"
	이렇게 두 생성 문장을 비교했을때, <font color="#ffff00">Candidate 1은 문장이 너무 짧아서 의미가 완전히 전달되지 않지만, N-gram precision 으로만 보면 Candidate2 보다 높은 점수</font>를 받게 됩니다.  이를 방지하기 위해, 문장이 Reference 보다 지나치게 짧을 경우 점수를 낮추는 패널티를 줍니다.
	
3. 최대 N-gram : 보통 1-gram 에서 4-gram 까지 고려하며, 각 N-gram 의 정밀도를 가중 평균합니다.

## Formula

BLEU 점수는 다음과 같이 계산됩니다.

#### n-gram 정밀도

- $p_n$ : n-gram 정확도
- $\text{Count}_{\text{match}}(g_n)$ : 참조 텍스트와 일치하는 n-gram의 개수
- $\text{Count}(g_n)$ : 생성된 텍스트에서 n-gram의 총 개수

$$
p_n = \frac{\Sigma_{C \in \text{candidate}} \Sigma_{g_n \in C}\text{Count}_{\text{match}}(g_n)}{\Sigma_{C \in \text{candidate}} \Sigma_{g_n \in C}\text{Count}(g_n)}
$$

## Examples

ROUGE와 비슷한 방식으로, BLEU는 n-gram 단위로 정밀도를 계산합니다. 예를 들어, 1-gram, 2-gram 등을 계산하며, 최종 점수는 이드르이 가중 평균으로 구합니다.

- 참조 텍스트(rs) : "인공지능은 컴퓨터가 학습하고 문제를 해결할 수 있게 합니다."
- 생성 텍스트(c) : "인공지능은 컴퓨터가 학습을 통해 발전합니다."

### Bleu-1 (1-gram)

- 참조 텍스트
	- 인공지능,은,컴퓨터가,학습하고,문제를,해결할,수,있게,합니다 = 9개

- 생성 텍스트의 1-gram:
    - 인공지능, 은, 컴퓨터가, 학습을, 통해, 발전합니다. = 6개
- 일치하는 1-gram:
    - 인공지능, 은, 컴퓨터가 = 3개

$$
P_1 = \frac{\text{Count}_{\text{match}}(g_1)}{\text{Count}(g_1)} = \frac{3}{6} = 0.5
$$

### Bleu-2 (2-gram)

- 참조 텍스트
	- 인공지능은, 은컴퓨터가, 컴퓨터가학습하고, 학습하고문제를, 문제를해결할, 해결할수, 수있게, 있게합니다. = 8개
- 생성 텍스트
	- 인공지능은, 은컴퓨터가, 컴퓨터가학습을, 학습을통해, 통해발전합니다. = 5개
- 일치하는 2-gram
	- 인공지능은, 은컴퓨터가 = 2개

$$
P_2 = \frac{\text{Count}_{\text{match}}(g_2)}{\text{Count}(g_2)} = \frac{2}{5} = 0.5
$$

위의 예시는<font color="#ffff00"> Brevity Penalty를 계산하지 않은 값</font>입니다.

