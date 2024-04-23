PretrainedConfig 는 [[transformers]] 라이브러리의 사전 학습 모델 구성을 나타내는 클래스입니다. 이 클래스는 모델의 [[Architecture]] , [[HyperParameter]] , 토큰화 관련 설정 및 다른 중요한 모델 구성을 포함합니다. 모델을 초기화할 때 이러한 구성을 사용하여 모델의 동작을 제어할 수 있습니다.

다음은 각 파라미터에 대한 설명입니다.

> name_or_path -> str, (optional), defaults to ""
- 모델의 사전 학습 체크포인트 구성 또는 구성을 식별하기 위한 문자열입니다. 

> output_hidden_state -> bool, (optional), defaults to False
- 모델이 모든 숨겨진 상태(hidden-states)를 반환해야 하는지 여부를 나타내는 불리언 값입니다. hidden state는 모델의 레이어 출력 중 하나입니다.

> output_attentions -> bool, (optional), defaults to False
- 모델이 모든 어텐션 출력(attention output)을 반환해야 하는지 여부를 나타내는 불리언 값입니다. <font color="#ffff00">어텐션 출력은 모델의 어텐션 가중치를 포함합니다.</font>

> return_dict -> bool, (optional), defaults to True
- 모델이 일반 튜플(tuple) 대신 ModelOutput을 반환해야 하는지 여부를 나타내는 불리언 값입니다. ModelOutput은 모델 출력의 이름 붙은 버전이며 <font color="#ffff00">사용자가 더 쉽게 결과를 검색할 수 있게 도와줍니다.</font>

> is_encoder_decoder -> bool, (optional), defaults to False
- 모델이 인코더/디코더로 사용되는지 여부를 나타내는 불리언값입니다. 이것은 번역 또는 요약과 같은 sequence-to-sequence 모델에서 사용됩니다.

> is_decoder -> bool, (optional), defaults to False
- 모델이 디코더로 사용되는지 여부를 나타내는 불리언 값입니다. 이것은 모델이 번역 또는 다른 디코더 역할을 수행하는 경우에 사용됩니다.

> cross_attention_hidden_size -> int, (optional)
- 모델이 인코더-디코더 구조에서 디코더로 사용될 때, cross-attention 레이어의 hidden dimension 크기를 나타내는 정수 값입니다. 이것은 `self.config.hidden_size` 와 다를 때 사용됩니다.

> add_cross_attention -> bool, (optional), defaults to False
- 모델에 cross attention 레이어를 추가해야 하는지 여부를 결정하는 불리언 값입니다. 이 파라미터는 <font color="ffff00">모델을 인코더와 디코더로 사용하는 경우</font>에 관련이 있으며, `EncoderDecoderModel` 클래스 내에서 사용됩니다. cross attention은 디코더가 인코더의 출력을 사용하는 경우에 유용합니다.

> tie_encoder_decoder -> bool, (optional), defaults to False
- 모든 인코더 가중치를 해당하는 디코더 가중치에 묶어야 하는지 여부를 결정하는 불리언 값입니다. <font color="ffff00">이것은 인코더와 디코더 모델이 정확히 동일한 매개변수 이름을 가져야 합니다. </font> 이러한 매개변수 묶음은 <font color="ffff00">가중치 공유를 위해 사용</font>됩니다.

> prune_heads -> Dict[int, List[int]], (optional), defaults to {}
- 모델의 head를 제거(prune) 해야 하는 경우 사용되는 파라미터입니다. 딕셔너리 형태로 제공되며, 특정 레이어의 head 인덱스를 제거합니다. 예를 들어 
```python
{1: [0, 2], 2: [2, 3]}
```

는 레이어 1에서 head 0 및, 2, 레이어 2에서 head 2 및 3을 제거합니다.

> chunk_size_feed_forward -> int, (optional), defaults to 0
- 모델의 모든 [[Feed Forward propagation]] 레이어의 청크 크기(chunk size)를 나타내는 파라미터입니다. 청크 크기는 피드 포워드 레이어가 한 번에 처리할 임베딩([[embedding]]) 개수를 결정합니다. 청크 <font color="#ffff00">크기가 0이면 피드 포워드 레이어가 청크로 분할되지 않습니다.</font> `n` 크기의 청크가 있으면 피드 포워드 레이어가 한 번에 `n` 보다 작은 시퀀스 길이의 임베딩을 처리합니다. 이는 피드 포워드 청크 처리에 대한 설정입니다.



