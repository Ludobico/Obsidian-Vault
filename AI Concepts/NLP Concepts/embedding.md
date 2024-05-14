Embedding이란, **텍스트를 실수 벡터 형태로 표현한 결과물을 의미**합니다. 아래 그림에서 보여주는 바와 같이, 특정한 단어, 문장 혹은 문서를 embedding 생성 모델에 입력하게 되면, 일정한 수의 실수들로 구성된 벡터가 출력됩니다.

![[Pasted image 20240514120919.png]]

Embedding을 사람이 직접 관찰하고 그 의미를 파악하기 어려우나, 서로 다른 단어 또는 문서로부터 추출된 embedding들 간의 거리를 계산하면 이들 간의 의미적 관계를 파악할 수 있습니다.

![[Pasted image 20240514120959.png]]

TensorFlow에서 제공하는 [Embedding Projector](https://projector.tensorflow.org/) 는 embedding에 대한 이해를 돕기 위해 만들어진 시각화 툴입니다. 여기에서는 Word2vec 이라는 embedding 방법을 적용하여 1만 개의 단어로부터 embedding들을 추출해 낸 뒤에, 이들을 3차원 공간 상에 투사한 결과를 볼 수 있습니다. 예를 드어, 아래 그림과 같이 공간 상에서 <font color="#ffff00">geographic</font> 이라는 단어의 embedding에 해당하는 점을 클릭할 경우, 공간 상에서 이와 가장 가까운 점들이 어느 단어의 embedding에 해당하는 지 볼 수 있습니다. 클릭한 단어와 실제로 의미적으로 유사한 단어들 (e.g geographical, coordinates, map, location 등)이 그 의미적 유사도를 기준으로 내림차순으로 나열되어 있는 것을 확인할 수 있습니다.

## Why embedding is required?
---

AI 모델은 기본적으로 하나의 function 이기 때문에, 기본적으로 숫자 형태의 입력만 받을 수 있고 숫자 형태의 결과만 출력할 수 있습니다. 그러나 사람이 입력한 텍스트의 경우 근본적으로 숫자가 아니기 때문에, 이를 AI 모델이 이해할 수 있는 숫자의 형태로 변형해 주어야 합니다. 이것이 embedding이 필요한 일차적인 이유입니다.

컴퓨터가 어떻게 작동하는지 잘 알고 계신 분이라면, 컴퓨터에서 다뤄지는 텍스트가 이미 숫자 형태로 인코딩(encoding)된 결과물인데 왜 다시 변형해 주어야 하는지 의아해하실 거라고 생각합니다.

embedding이 필요한 좀 더 중요한 이유는, 텍스트의 길이가 일반적으로 매우 길고 그 길이의 가변셩 또한 매우 크다는 것입니다. AI 모델의 내부 구조 상 이렇게 길이가 길고 가변적인 입력값을 다루는 데 특화되어 있지 않기 때문에, 숫자만으로 구성된 고정적인 길이의 입력값인 embedding으로 기존 테스트를 변환하여 AI 모델에게 전달해 주어야 합니다.

## Word (token) embedding vs Sentence/Document embedding
---

Embedding을 추출할 원본 텍스트의 형태를 기준으로 embedding을 구분하자면, 크게 Word(token) embedding과 Sentence/Document embedidng으로 구분할 수 있습니다.

원본 텍스트는 AI 모델에 입력되기 전에 더 작은 조각들로 쪼개지는 과정을 반드시 먼저 거칩니다. 이 때의 **조각을 [[token]]**, 쪼개는 일을 하는 모델을 [[Tokenizer]] 라고 자칭하며, 어떤 tokenizer를 사용하느냐에 따라 하나의 token이 곧 "word" 하나가 될 수 있고 subword(단어의 일부 조각)가 될수도 있습니다. 이런 token 하나로부터 추출한 embedding을 흔히 Word(token) embedding 이라고 부릅니다.

![[Pasted image 20240514122639.png]]

OpenAI에서 제공하는 [GPT-3 Tokenizer](https://platform.openai.com/tokenizer) 는, GPT 계열의 모델에서 사용되는 tokenizer가 어떻게 동작하는지를 이해하기 쉽게 보여주는 툴입니다. 예를 들어 아래 그림과 같이 해당툴의 입력란에 원하는 영문 텍스트를 입력하면, 해당 텍스트가 어떠한 (subword) tokens로 쪼개져서 AI 모델에 들어가게 되는지 한 눈에 확인할 수 있습니다.

한편, 사용자가 입력으로 넣기를 원하는 것은 보통 단어 하나가 아니라 하나의 문장 또는 여러 문장으로 구성된 하나의 문서일 텐데, 이렇게 문장 또는 문서 전체로부터 추출한 embedding을 **Sentence embedding** 또는 **Document embedding** 이라고 부릅니다. Sentence/Document embedding의 경우, 이를 구성하는 (sub)word 들로부터 여러 개의 word(token) embedding들을, averaging(평균) 연산 등을 통해 하나의 embedding으로 집계하는 방식으로 얻어집니다.

## How to create embedding?
---

### In the past  : One-hot Encoding

![[Pasted image 20240514123158.png]]

Embedding을 만들기 위해, 과거에는 비교적 단순한 방식을 채택했습니다. AI 모델을 학습하기 위한 거대한 문서 모음집이 있다고 가정했을 때, 먼저 해당 모음집에 등장하는 모든 단어들을 카운팅하고, 이들을 가지고 거대한 단어집을 하나 만든 뒤, 단어집 내의 각 단어에 대해 숫자 인덱스를 하나씩 부여합니다. 이를 바탕으로 하여, 어떤 단어가 제시되었을 때, 단어집 내에 포함된 전체 단어 수만큼의 길이를 가지는 숫자 0으로만 구성된 벡터를 하나 만들고, 제시된 단어의 숫자 인덱스에 해당하는 위치에만 1을 배치하여 최종적인 embedding vector를 생성합니다. 이러한 embedding 생성 방법을 **One-Hot encoding** 이라고 합니다.

One-Hot Encoding의 경우 그 자체로 사람이 보고 이해하기에는 용이하나, 단어 수에 따라 embedding vector의 차원이 지나치게 커지는 경향이 있고(100k ~ 1M), embedding vector 내 절대 다수의 원소가 0 값을 가지는 극도로 성긴(sparse) 구성을 가지고 있어서 AI 모델이 효과적으로 해늘링하기에 어려운 측면이 있습니다.

### Current : Learned Embedding

![[Pasted image 20240514123756.png]]

반면, **Learned Embedding**의 경우 위에서 서술한 One-Hot encoding의 단점을 보완하고자 새로 등장한 embedding 생성 방법으로, 본 글에서 언급하는 embedding이 바로 이 Learned Embedding에 해당합니다. 일반적으로 거대 문서 모음집을 사용하여 Neural Network 구조를 지니는 AI 모델 또는 LLM을 학습하여 Learned Embedding 생성 모델이 얻어지는데, 이 학습 과정에서 AI 모델은 다양한 단어들을 받아들이게 되고, 그 과정에서 **문맥 상 의미적으로 유사한 단어들의 embedding vector들 간의 거리는 가깝게, 그렇지 않은 단어들의 embedding vector들 간의 거리는 멀게** 만드는 방식으로 의미적 관계를 학습하게 됩니다.

One-Hot encoding 과는 반대로, Learned Embedding의 경우 사람이 보고 이해하기는 어려운 특성을 지닙니다. 그러나 One-Hot encoding 방식에 비해 embedding vector의 차원이 매우 낮고(384 ~ 1,536), embedding vector 내 모든 원소를 밀도 있게 활용하고 있어서 AI 모델이 이를 보다 효과적으로 핸들링할 수 있다는 장점이 있습니다.

## OpenAI Embeddings (GPT-3)
---

텍스트로부터 효과적인 embedding을 추출하기 위해 현재 전 세계적으로 가장 많이 사용되는 수단으로 OpenAI Embeddings API가 있습니다. 거대한 LLM을 구동하는 데 필요한 컴퓨팅 리소스를 보유하고 있을 필요 없이, 적은 비용을 지불하여 아래 예시와 같이 단순히 몇 줄로 구성된 Python 코드를 통해 API 호출을 하는 방식으로 embedding을 추출할 수 있어 많은 개발자들의 사랑을 받고 있습니다.

![[Pasted image 20240514124257.png]]

