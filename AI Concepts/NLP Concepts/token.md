<font color="#ffff00">token</font> 이라는 용어는 보통 자연어 처리(NLP) 분야에서 많이 사용되는 개념입니다. 이는 문장을 구성하는 단위로 사용되며, 주로 단어, 부분 단어(sub word), 또는 문장부호를 포함합니다. 간단히 말해서, **문장을 구성하는 개별 요소**를 말합니다.

예를 들어, 문장 "I love ice-cream"을 토큰화하면 \["I", "love", "ice-cream"\] 으로 분리될 수 있습니다. 여기서 각각의 요소 "I", "love", "ice-cream" 이 하나의 토큰으로 처리됩니다.

subword 토큰화는 주로 일반적이지 않은 단어나 긴 단어를 더 작은 단위로 분할할 때 사용됩니다. 예를 들어 "antidisestablishmentarianism" 라는 긴 단어를 \["anti", "dis", "establishment", "arian", "ism"\] 과 같이 여러 subwords로 나눌 수 있습니다. 이런 방식은 단어의 내부 구조를 이해하고, 훈련 데이터에 자주 나타나지 않은 단어들에 대한 처리를 개선하는데 도움이 됩니다.

토큰화는 자연어 처리에서 매우 중요한 단계로, 텍스트를 분석하거나 처리하기 전에 실시하는 초기단계 중 하나입니다. 토큰화를 통해 얻어진 토큰들은 후속 처리 단계에서 입력 데이터로 사용됩니다.