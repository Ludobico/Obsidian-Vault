
`max_tokens` 는 **모델이 생성하는 출력의 최대 길이를 의미**합니다. 입력 토큰과는 무관하게, 생성된 응답이 일정 길이를 초과하지 않도록 제한하기 위해 사용됩니다. 예를 들어 질문이 50토큰이고 `max_tokens` 를 100으로 설정하면, 응답은 최대 100토큰까지만 생성되며 전체는 150토큰으로 계산됩니다. 이때도 [[context length]] 를 초과하면 안됩니다.

