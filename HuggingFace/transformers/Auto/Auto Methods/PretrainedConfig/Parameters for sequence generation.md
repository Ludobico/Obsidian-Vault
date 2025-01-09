> max_length -> int, (optional), defaults to 20
- 모델의 `generate` 메서드에서 기본적으로 사용되는 최대 길이를 나타냅니다. 이 값은 <font color="#ffff00">생성된 시퀀스의 최대 길이를 제한하는데 사용</font>됩니다. 생성된 시퀀슥 이 최대 길이를 초과하면 잘라질 수 있습니다.

> min)length -> int, (optional), defaults to 0
- 모델의 `generate` 메서드에서 기본적으로 사용되는 최소 길이를 나타냅니다. 이 값은 <font color="#ffff00">생성된 시퀀스의 최소 길이를 제한</font>하는 데 사용됩니다. 생성된 시퀀스가 이 최소 길이 미만이면 패딩([[padding]])이 추가될 수 있습니다.

> do_sample -> bool, (optional), defaults to False
- 모델의 `generate` 메서드에서 기본적으로 사용되는 플래그입니다. 이 플래그가 <font color="#ffc000">True</font> 이면 샘플링을 사용하며, False 이면 greedy decoding이 사용됩니다. <font color="#ffff00">샘플링을 사용하면 생성된 시퀀스가 더 다양</font>해집니다.

> early_stopping -> bool, (optional), defaults to False
- 이 플래그가 True이면 beam search 중 하나의 배치에서 적어도 `num_beams` 개의 문장이 완료되면 검색을 중단합니다.

> num_beams -> int, (optional), defaults to 1
- beam search의 빔 수를 지정합니다. beam search를 사용하면 여러 가설이 고려되어 다양한 시퀀스가 생성됩니다. `num_beams` <font color="#ffff00">값이 1이면 beam search를 사용하지 않고 단일 가설만 고려</font>됩니다.

> num_beam_groups -> int, (optional), defaults to 1
- beam search 중에서 다양성을 보장하기 위해 `num_beams`를 여러 그룹으로 나누는 데 사용됩니다. 이 파라미터는 group beam search를 활성화하는 역할을 합니다. `num_beam_group` 값이 1이면 group beam search를 사용하지 않고 모든 빔이 하나의 그룹으로 간주됩니다.

> diversity_penaly -> float, (optional), defaults to 0.0
- 다양성을 제어하는 데 사용됩니다. 값이 0.0이면 다양성 패널티가 없고, 더 높은 값은 출력의 다양성을 촉진합니다. 값이 높을수록 모델이 더 다양한 토큰을 선택하려고 시도하며 출력이 더 다양해집니다.

> [[temperature]] -> float, (optional), defaults to 1.0
- 다음 토큰 확률을 모델링하는 데 사용되는 [[temperature]]를 제어합니다. 온도가 크면 확률 분포가 더 평탄해져 무작위성이 증가하고, 온도가 낮으면 모델이 더 확신 있는 토큰을 선택합니다. 온도 값은 양수여야 합니다.

> [[top_k]] -> int, (optional), defaults to 50
- 상위 k개의 필터링을 나타내며, 가장 높은 확률을 가진 상위 `k` 개의 어휘 토큰을 유지합니다. 이를 통해 생성된 시퀀스의 다양성을 조절할 수 있습니다.

> [[top_p]] -> float, (optional), defaults to 1
- top-p 필터링을 나타내며, 확률 합이 `top_p` 이상인 가장 확률이 높은 토큰만을 유지합니다. `top_p` 값은 1보다 작은 부동소수점 값이면서 양수이며, 시퀀스 생성의 품질을 높이는 데 사용됩니다.

> typical_p -> float, (optional), defaults to 1
- <font color="#ffc000">typicality</font> 는 시퀀스의 가장 일반적인 토큰을 선택하는데 사용되는 파라미터입니다. 이 파라미터를 설정하면, 조건부 확률로 예측된 특정 토큰이 다른 무작위 토큰을 예측하는 것과 얼마나 유사한지를 측정합니다. `typical_p` 값이 1보다 작은 경우, 확률이 가장 높은 일반적인 토큰만 유지됩니다. 일반적으로 토큰의 일관성을 유지하고 반복을 줄이는 데 사용됩니다.

> repetition_penalty -> float, (optional), defaults to 1
- 반복 패널티는 생성된 시퀀스에서 <font color="#ffff00">동일한 토큰이 반복되는 것을 억제하기 위해 사용</font>됩니다. 값이 1.0 이면 패널티가 없으며, 값이 높을수록 반복이 더 강하게 억제됩니다.

> length_penalty -> float, (optional), defaults to 1
- 길이 패널티는 beam based generation에 사용되며, 시퀀스 길이에 지수적인 패널티를 부과합니다. 길이 패널티는 시퀀스의 길이로 스코어를 나누는데 적용되며, 길이 패널티 값이 큰 경우 더 긴 시퀀스를 장려하고, 값이 작은 경우 더 짧은 시퀀스를 장려합니다. 이는 생성된 시퀀스의 길이를 조절하는 데 사용됩니다.

> no_repeat_ngram_size -> int, (optional), defaults to 0
- 이 파라미터는 반복되는 n-그램(n-grams) 시퀀스를 방지하는 데 사용됩니다. 값이 0보다 큰 경우, 특정 크기의 n-그램은 단 한번만 발생할 수 있습니다.

> encoder_no_repeat_ngram_size -> int, (optional), defaults to 0
- 이 파라미터는 인코더와 디코더 사이에 사용되며 인코더에서 발생한 특정 크기의 n-그램이 디코더에서 발생하는 것을 방지합니다. 마찬가지로 값이 0보다 큰 경우, 해당 크기이 n-긂은 두 곳에서 모두 나타나지 않습니다.

> bad_words_ids -> List[int], (optional)
- 이 파라미터는 생성된 텍스트에서 허용되지 않는 단어나 토큰의 ID 목록입니다. 기<font color="#ffff00">본적으로 지정된 단어나 토큰이 생성된 텍스트에 나타나지 않도록 막을 때 사용</font>됩니다.

> num_return_sequences -> int, (optional), defaults to 1
- 이 파라미터는 각 배치 요소에 대해 생성된 시퀀스의 개수를 나타냅니다. 기본적으로 1개의 시퀀스만 반환되지만, 원하는 경우 더 많은 시퀀스를 반환할 수 있습니다.

> output_score -> bool, (optional), defaults to False
- 이 파라미터는 모델이 생성할 때 로짓(scores)을 반환할지 여부를 제어합니다. 만약 <font color="#ffc000">True</font>로 설정하면, 생성 과정에서 발생한 로짓이 반환됩니다.

> return_dict_in_generate -> bool, (optional), defaults to False
- 이 파라미터는 모델이 생성할 때 `modelOutput` 대신 `torch.LongTensor`를 반환해야 하는지를 제어합니다. `True`로 설정하면 모델의 출력이 `modelOutput` 객첼 반환됩니다. 이 객체는 생성 시에 추가정보와 출력을 함께 제공합니다.

> forced_bos_token_id -> int, (optional)
- 이 파라미터는 디코딩 시작 토큰인 `decoder_start_token_id` 이후에 생성되는 첫 번째 토큰을 강제로 지정합니다. 특히, 다국어 모델인 경우 원하는 대상 언어 토큰을 첫 번째로 생성하기 위해 사용될 수 있습니다.

> forced_eos_token_id -> int, (optional)
- 이 파라미터는 `max_length`에 도달했을 때 강제로 생성되는 마지막 토큰을 지정합니다. 생성된 시퀀스의 길이가 `max_length`에 도달하면 이토큰이 마지막으로 추가됩니다.

> remove_invalid_values -> bool, (optional)
- 이 파라미터는 모델의 출력에서 가능한 `nan` 및 `inf` 값을 제거하여 생성 과정을 중단시키지 않도록 제어합니다. 그러나 `remove_invalid_values` 를 사용하면 생성 과정이 느려질 수 있음에 주의해야 합니다. 이것은 모델의 안정성을 확보하는 데 도움이 될 수 있습니다.

