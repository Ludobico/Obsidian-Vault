- [[#Understanding through example|Understanding through example]]
	- [[#Understanding through example#num_beams = 1|num_beams = 1]]
	- [[#Understanding through example#num_beams = 4|num_beams = 4]]
	- [[#Understanding through example#num_beams = 10|num_beams = 10]]
- [[#Recommended settings|Recommended settings]]


`num_beams` 는 텍스트 생성에서 **빔 서치(beam search) 알고리즘의 하이퍼파라미터**로, 탐색할 시퀀스(경로)의 수를 결정합니다. 빔 서치는 언어 모델이 다음 토큰을 예측할 때 여러 가능성을 병렬적으로 고려해서 가장 가능성 높은 텍스트 시퀀스를 선택하는 방법입니다.

![[Pasted image 20250411094031.png]]

빔 서치는 확률 기반으로 시퀀스를 생성하며, `num_beams` 는 유지할 시퀀스 수를 정의합니다. 수학적으로 표현하면 다음과 같습니다.

- 언어 모델은 각 단계 $t$ 에서 다음 토큰 $w_t$ 의 조건부 확률을 계산

$$
P(w_t|w_1, w_2, ..., w_{t-1}, \text{context})
$$

여기서 $w_1, w_2, ..., w_{t-1}$ 은 이전 토큰 시퀀스, context는 입력(input)값입니다.

- 각 시퀀스의 점수는 로그 확률의 합으로 계산합니다.

$$
\text{Score}(w_1, w_2, ..., w_t) = \Sigma^t_{i=1} \log p(w_i | w_1, ..., w_{i-1}, \text{context})
$$

## Understanding through example

빔 서치는 트리 구조로 시각화할 수 있습니다. 각 노드는 토큰, 각 경로는 시퀀스를 나타냅니다.

> The image shows a

- 어휘 : \["dog", "cat", "bird", ... \]
- `num_beams = 2`

1. 첫 번째 단계
	- "The image shows a dog" (점수 : $\log P(\text{dog})$)
	- "The image shows a cat" (점수 : $\log P(\text{cat})$)

2. 두 번째 단계
	- "The image shows a dog running" ($\log P(\text{dog}) + \log P(\text{running})$)
	- "The image shows a cat sitting ($\log P(\text{cat}) + \log P(\text{sitting})$)"
	- 다시 상위 2개 유지

```
The image shows a
├── dog (0.4) ── running (0.3) ── fast (0.2)
├── cat (0.3) ── sitting (0.25) ── quietly (0.15)
└── bird (0.2) [제외됨]

num_beams=2: dog/cat 경로 유지
```

### num_beams = 1
- Greedy search
- 가장 높은 확률의 토큰만 선택하므로 간단하지만 부정확하거나 덜 자연스러움

### num_beams = 4
- 여러 시퀀스를 탐색해 더 자연스럽고 상세한 답변 생성

### num_beams = 10
- 더 많은 경로를 고려해 세련된 답변이 가능하지만, **VRAM 사용량 증가**
## Recommended settings

- 일반 : 4~8
- 의료 질문 : 5~6
- 저사양 GPU : 2~3

