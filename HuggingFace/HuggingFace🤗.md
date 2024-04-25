![[Pasted image 20231019101856.png]]

HuggingFace🤗는 <font color="#ffff00">자연어처리를 위한 오픈 소스 플랫폼과 라이브러리를 개발하고 유지 및 관리하는 회사</font>입니다. HuggingFace🤗는 NLP모델, [[Tokenizer]], 사전 훈련된 모델, 데이터셋 등을 제공하여 개발자들이 NLP 작업을 보다 쉽고 효율적으로 수행할 수 있도록 지원합니다.

## Installation
---
HuggingFace🤗 를 설치하기에 앞서 아래와 같은 요구사항이 필요합니다.
- 3.6 버전 이상의 파이썬
- 1.1.0 버전 이상의 [[Pytorch]]

```bash
pip install transformers
```

모듈을 불러올때 HuggingFace가 아닌 [[transformers]] 를 불러옵니다.
## datasets
---
datasets 라이브러리는 HuggingFace🤗 에서 제공하는 라이브러리입니다. 이 라이브러리는 <font color="#ffff00">다양한 자연어 처리 데이터셋에 대한 액세스와 데이터 준비를 단순화하는데 도움을 주는 도구를 제공</font>합니다. datasets 라이브러리를 사용하면 다양한 자연어 처리 작업에서 데이터를 쉽게 불러오고 전처리할 수 있으며, 모델 훈련 및 평가에 사용할 수 있는 데이터셋을 쉽게 다운로드 미 관리할 수 있습니다.

```bash
pip install datasets
```

