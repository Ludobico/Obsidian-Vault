
`DataCollatorForLanguageModeling` 은 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 제공하는 [[DataCollator]]로, 언어 모델링 작업(특히 파인튜닝)을 위해 **데이터 배치를 준비**하는 역할을 합니다.

데이터 콜레이터는 데이터셋의 샘플들을 배치로 묶을 때, 모델이 학습할 수 있도록 입력 데이터를 적절히 전처리하고 포맷팅하는 역할을 합니다. `DataCollatorForLanguageModeling` 의 주요기능은 다음과 같습니다.

<font color="#ffff00">토큰 패딩 및 마스킹 처리</font>
- 배치 내 샘ㅍ믈들의 길이가 다를 수 있으므로 동일한 길이로 [[padding]] 을 추가합니다.
- `mlm=False` 설정 시, Causal Language Model(CLM) 을 위해 입력 데이터와 레잉블을 준비합니다. 즉, 입력 시퀀스 자체를 예측 대상으로 설정하고, 패딩 토큰을 loss function 에서 제외되도록 마스크를 생성합니다.

<font color="#ffff00">레이블 생성</font>
- CLM의 경우, 모델은 다음 토큰을 예측하도록 학습합니다. 데이터 콜레이터는 입력 시퀀스를 그대로 레이블로 사용하되, 패딩된 부분은 loss function 에서 제외도도록 처리합니다.

<font color="#ffff00">배치 정규화</font>
- 여러 샘플을 하나의 텐서로 묶어 모델에 전달할 수 있도록 포맷을 맞춥니다.

<font color="#ffff00">mlm=False</font>
- `mlm=False` 는 Masked Language Model(MLM, 예 : Bert)이 아니라 Causal Language Model(CLM, 예 : GPT)을 수행하도록 설정합니다.

