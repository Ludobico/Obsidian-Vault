---
_filters: []
_contexts: []
_links: []
_sort:
  field: rank
  asc: false
  group: false
---
![[Pasted image 20240426170850.png]]

[[HuggingFace🤗]] 의 TRL(Transformer Reinforcement learning) 은 **강화학습을 통해 [[Transformer]] 언어 모델을 학습시키기 위한 풀스택 라이브러리**입니다. 주요 모델은 다음과 같습니다.

## SFT(Supervised Fine-tuning)
---
기존 언어 모델을 특정 태스크에 맞게 파인튜닝 합니다.

## RM(Reward Modeling)
---
모델의 출력에 대한 보상(reward) 함수를 학습시킵니다.

## PPO(Proxial Policy Optimization)
---
강화학습 알고리즘인 [[PPO(Proximal Policy Optimization)]] 를 사용하여 보상 함수를 최대화하는 방향으로 모델을 업데이트합니다.

