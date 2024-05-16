[원본 링크](https://www.syncly.kr/blog/what-is-embedding-and-how-to-use)

- [[#Why embedding is required?|Why embedding is required?]]
- [[#Word (token) embedding vs Sentence/Document embedding|Word (token) embedding vs Sentence/Document embedding]]
- [[#How to create embedding?|How to create embedding?]]
	- [[#How to create embedding?#In the past  : One-hot Encoding|In the past  : One-hot Encoding]]
	- [[#How to create embedding?#Current : Learned Embedding|Current : Learned Embedding]]
- [[#OpenAI Embeddings (GPT-3)|OpenAI Embeddings (GPT-3)]]
- [[#When do we use Embedding?|When do we use Embedding?]]
- [[#Semantic Search|Semantic Search]]
- [[#Recommendation|Recommendation]]
- [[#Clustering|Clustering]]
- [[#Vector Database|Vector Database]]


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

OpenAI 측에서 제공한 가이드 문서에 의거하면, OpenAI Embeddings에서는 embedding 추출을 위해 GPT-3 계열의 LLM을 사용합니다. OpenAI Embeddings로부터 추출한 embedding vector는 (text-embedding-ada-002 모델) 1,536차원이기 때문에, 길이가 긴 텍스트에 담긴 의미적 정보 또한 충실하게 담기에 적절한 구성을 가지고 있다고 할 수 있습니다.

## When do we use Embedding?
---

Embedding을 사용하면 서로 다른 단어 또는 문서들 간의 의미적 관계를 파악할 수 있다고 하였는데, 이러한 특성이 유용하게 적용될 수 있는 몇 가지 대표적이 케이스들이 있습니다. 이들을 크게 아래에 두 가지 카테고리로 분류할 수 있습니다.

첫 번째 케이스는, **여러 문서들이 존재할 때 이들 중 하나를 탐색**하거나, **이들을 서로 비교해야 하는 경우**입니다. 대표적으로 <font color="#ffff00">Semantic Search</font> , <font color="#ffff00">Recommendation</font>, <font color="#ffff00">Clustering</font> 등의 기능이 이에 해당합니다.

OpenAI의 Embeddings를 사용하는 경우, OpenAI Cookbook에서 제공하는 [Semantic text search using embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb) , [Recommendation using embeddings and nearest neighbor search](https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb) , [Clustering](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb) 노트북을 보면, 각 기능을 [[Python]] 코드로 어떻게 구현할 수 있는지 상세히 확인할 수 있습니다.

## Semantic Search
---
![[Pasted image 20240516093413.png]]

Semantic Search는 **사용자가 제시한 텍스트 형태의 query와 의미적으로 연관성이 높은 문서들을 찾아서 제시**해 주는 기능입니다. Embeddings를 활용한 Semantic Search 프로세스를 간략하게 정리하면 아래와 같습니다.

1. 문서 모음집에 포함되어 있는 각각의 문서에 대한 embedding을 계산하여 별도의 저장소 (e.g local drive, vector database 등)에 저장
2. Query 텍스트에 대한 embedding을 계산
3. Query embedding과 각 문서 embedding 간의 cosine similarity(코사인 유사도)를 계산하고, 그 값을 기준으로 전체 문서들을 내림차순으로 정렬
4. 정렬 결과 중 상위 $k$개에 해당하는 문서들의 텍스트를 불러온 뒤 이를 반환함

## Recommendation
---
![[Pasted image 20240516093832.png]]

Recommendation은 **사용자가 현재 보고 있는 문서와 의미적으로 연관성이 높은 다른 문서들을 찾아서 제시**해 주는 기능입니다. Recommendation 프로세스는 Sementic Search 프로세스와 거의 동일하나, query embedding이 현재 보고 있는 문서의 embedding으로 그대로 대체된 것이라고 생각하면 됩니다.

1.  문서 모음집에 포함되어 있는 각각의 문서에 대한 embedding을 계산하여 별도의 저장소 (e.g local drive, vector database 등)에 저장
2. 현재 보고 있는 문서의 embedding과 그 외의 문서들 각각의 embedding 간의 cosine similarity(코사인 유사도)를 계한하고, 그 값을 기준으로 전체 문서들을 내림차순으로 정렬
3. 정렬 결과 상위 $k$ 개에 해당하는 문서들의 텍스트를 불러온 뒤 이를 반환함

## Clustering
---

![[Pasted image 20240516100317.png]]

Clustering은 **여러 문서들 간의 의미적 유사성을 바탕으로 이들을 몇 개의 그룹으로 묶어서 정리**해주는 기능입니다. Semantic Search와의 주요한 차이가 있다면, 많은 수의 문서 쌍들이 embedding 간의 거리를 계산해야 한다는 점입니다.

Python의 경우 scikit-learn과 같은 ML 라이브러리를 같이 사용한다면, 입력값으로 embedding vector들을 대신 넣어주기만 하면, Clustering 알고리즘에 대해 자세히 알지 못하더라도 손쉽게 Clustering을 수행할 수 있습니다.


두 번째 케이스는 좀 더 복잡한 경우인데, 이는 현재 통용되고 있는 LLM의 근본적인 특성과 관련이 있습니다. LLM은 인터넷 상에서 검색 가능한 공개된 정보에 대한 일반적인 지식을 가지고 있으나, 여러분이 가지고 있을 수 있는 비공개 정보에 대한 지식을 가지고 있지 않습니다. 따라서 여러분이 LLM으로 하여금 여러분만이 가지고 있는 정보와 연관된 결과물을 기반으로 어떤 질문에 대한 답을 출력하고자 한다면, 해당 정보를 담은 텍스트를 LLM의 prompt에 함께 포함시켜 요청해야 합니다.

그러나 현재 서비스되고 있는 LLM 서비스들의 경우 prompt에 추가될 수 있는 텍스트의 길이(i.e. token의 총 개수)가 제한되어 있기 때문에, 만약 여러분이 제시하고자 하는 텍스트의 길이가 책 한 권 수준으로 매우 길다면, 이를 미리 여러 chunk(덩어리)로 쪼개 놓고 이들 중 주어진 질문과 가장 연관성이 높은 chunk만을 골라서 prompt에 추가해야 합니다.

![[Pasted image 20240516100945.png]]

이는 특히 **Question Answering 과 같은 기능을 구현하고자 할 때 흔히 고려하게 되는 포인트**인데, LLM에게 추가로 주입된 정보를 바탕으로 주어진 질문(query)에 대한 답변을 수행하도록 하는 일반적인 Q&A 프로세스를 간략하게 정리하면 아래와 같습니다. 이는 Semantic Search 프로세스와 어느 정도 유사합니다.

1. 전체 정보를 담은 전체 텍스트를 일정한 길이로 분할하여 여러 개의 텍스트 chunk를 구성하고, 각각의 chunk에 대한 embedding을 계산하여 별도의 저장소에 저장
2. 질문 내용을 담은 query 텍스트에 대한 embedding을 계산
3. Query embedding과 각 chunk embedding 간의 cosine similarity를 계산하고, 그 값을 기준으로 전체 문서들을 내림차순 정렬
4. 정렬 결과 중 상위 $k$ 개에 해당하는 문서들의 텍스트를 불러온 뒤, 이를 prompt에 추가 - 이 때의 $k$ 는 LLM 서비스에서 요구하는 최대 텍스트의 길이 제약 하에서의 가능한 최댓값으로 결정
5. 완성된 prompt를 LLM에 입력하고, 이에 대한 LLM의 생성된 답변을 반환

OpenAI Embeddings를 사용하는 경우, OpenAI Cookboo에서 제공하는 [Qestion answering using embeddings-based search](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb) 노트북을 보면, 해당 기능을 Python 코드로 어떻게 구현할 수 있는지 상세히 확인할 수 있습니다.

## Vector Database
---

LLM으로 하여금 원하는 결과를 생성해 내기 위해서 추가적인 정보가 필요한 경우 바로 위에서 서술한 과정과 같이 embedding을 활용해서 하면 되는데, LLM이 한 번 입력받은 정보를 "기억"하지 못하기때문에, 결과를 생성하고자 할 때마다 이러한 과정을 매번 반복해야 한다는 한계점을 가지고 있습니다. 이를 흔히 stateless(상태가 저장되지 않는)한 특성이라고 부르며, 이는 보다 복잡한 작업을 수행하도록 명령하는 데에 있어서 다소 불편함으로 작용합니다.

![[Pasted image 20240516101827.png]]

<font color="#ffff00">Vector Database</font> 는 LLLM을 비롯한 AI 모델의 이와 같은 **장기 기억 능력의 부재를 보완하기 위해 나온 새로윤 유형의 데이터베이스**입니다. RDBMS와 같은 전통적인 데이터베이스와는 다르게 Vector Database는 고차원의 실수 벡터 인덱스를 효율적으로 저장하는 데 특화되어 있습니다. 또한, SQL 등으로 표현된 query에 대하여 이와 정확하게 매칭되는 결과물을 추출하는 것이 아니라, query 또한 embedding의 형태로 표현되어 있고, 해당 query embedding 과의 유사도가 가장 높은 embedding을 가지는 데이터를 추출하는 방식을 지원합니다. 즉, AI 모델로부터 얻어진 embedding에 대한 저장 및 이를 활용한 데이터 추출 등에 최적화되어 있다고 할 수 있습니다.

만약 여러분들이 LLM으로 하여금 수행하고자 하는 작업이 고차원(e.g 512차원 이상)의 embedding을 요구하고, 다루게 되는 전체 embedding의 개수 또한 많다면 (e.g 10,000개 이상) vector database를 사용하는 걸 적극 고려해 보시길 바랍니다. 즉각적으로 사용해 볼 수 있는 오픈 소스 vector Database로 Chroma가 있는데, OpenAI cookbook에서 제공하는 [Robust Question Ansering with Chroma and OpenAI](https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/chroma/hyde-with-chroma-and-openai.ipynb) 노트북을 보면, Question Answering 기능을 Chroma를 사용하여 구현하는 과정을 상세히 확인할 수 있습니다.