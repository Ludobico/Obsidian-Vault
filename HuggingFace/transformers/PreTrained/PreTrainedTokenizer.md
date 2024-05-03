
[[transformers]] 의 `PreTrainedTokenizer` 클래스는 [[Tokenizer]] 를 초기화할 때 다양한 매개변수를 설정할 수 있도록 설계되었습니다. 이러한 매개변수들은 토크나이저의 동작 방식과 결과에 큰 영향을 미칠 수 있습니다. 다음은 주요 매개변수들에 대한 설명입니다.

> model_max_length -> int, optional
- [[Transformer]] 모델의 입력으로 사용되는 최대 토큰 수를 정의합니다.
- [[from_pretrained]] 를 통해 토크나이저를 로드할 때, 관련 모델에 저장된 `max_model_input_sizes` 값을 사용하여 설정됩니다.
- 기본값은  매우 큰 정수 `int(1e30)`로 설정됩니다.

> padding_side -> str, optional
- 패딩이 적용될 위치를 정의합니다. `right` `left` 중에서 선택할 수 있습니다.
- 기본값은 클래스 속성에서 가져옵니다.

> truncation_side -> str, optional
- 입력이 최대 길이를 초과하는 경우 잘라낼 위치를 정의합니다. `right` `left` 중에서 선택할 수 있습니다.
- 기본값은 클래스 속성에서 가져옵니다.

> chat_template -> str, optional
- 채팅 메시지 목록을 형식화하는 데 사용되는 `jinja` 템플릿 문자열입니다.
- 자세한 설명은 [[HuggingFace🤗]] 문서에서 확인할 수 있습니다.

> model_input_names -> List[str], optional
- 모델의 포워드 패스에서 허용되는 input list입니다. 예("token_type_ids", "attention_mask")
- 기본값은 클래스 속성에서 가져옵니다.

> bos_token -> str, [[Tokenizer]] , AddedToken, optional
- 문장의 시작을 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> eos_token -> str, [[Tokenizer]] , AddedToken, optional
- 문장의 끝을 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> unk_token -> str, [[Tokenizer]] , AddedToken, optional
- 알려지지 않은 문장을 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> sep_token -> str, [[Tokenizer]] , AddedToken, optional
- 문장의 분리를 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> pad_token -> str, [[Tokenizer]] , AddedToken, optional
- [[padding]]을 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> cls_token -> str, [[Tokenizer]] , AddedToken, optional
- 입력 클래스를 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> mask_token -> str, [[Tokenizer]] , AddedToken, optional
- 마스크된 토큰을 나타내는 특수 토큰입니다.
- 이 토큰은 해당 토크나이저 속성(`bos_token_id`, `eos_token_id` 등)과 연관됩니다.

> additional_special_tokens -> tuple or list of str or [[Tokenizer]] , AddedToken, optional
- 추가적인 특수 토큰의 리스트입니다. 디코딩 시 특수 토큰을 건너뛰도록 서렂ㅇ하려면 여기에 추가하세요.
- 어휘에 포함되지 않은 경우 어휘의 끝에 추가됩니다.

> clean_up_tokenization_speces -> bool, optional, Default : True
- 토크나이징 과정에서 추가된 공백을 처리할지 여부를 결정합니다.

> spilit_special_tokens -> bool, optional, Default False
- 특수 토큰을 분할할지 여부를 설정합니다.
- 기본적으로 특수 토큰은 분할되지 않습니다. 예를 들어, `<s>` 가 `bos_token` 인 경우, `split_special_tokens=False` 일 때 `tokenizer.tokenize(<s>)` 는 `['<s>']` 를 반환합니다. 반면 `True` 일 경우, `['<', 's', '>']` 를 반환합니다. 

