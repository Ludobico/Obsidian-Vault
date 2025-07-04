- [[#Parameters|Parameters]]

## Parameters

> gpt_batch_size -> int

[[Autoregressive]] 생성 시 한 번에 처리할 샘플의 수를 정의합니다.

> enable_redaction -> bool, default : True

민감한 데이터를 제거하거나 비식별화(redaction)를 활성화할지 여부를 설정합니다.

> kv_cache -> bool, default : True

key-value 캐싱 사용 여부를 결정합니다.
key-value 캐싱은 이전 연산 결과를 저장해 계산 속도를 향상시킵니다.

> get_checkpoint -> str

오토리그레시브 모델의 체크포인트 파일 경로를 지정합니다.

> clvp_checkpoint -> str

CLVP 모델 (Conditional Latent Variable Perseq) 의 체크포인트 경로를 지정합니다.
CLVP은 **입력 텍스트와 음성을 매칭하여 조건부 출력을 생성**하는 데 사용됩니다.

> decoder_checkpoint -> str

DiffTTS 디코더 모델의 체크포인트 경로를 지정합니다.
DiffTTS는 음성의 세부적인 특징을 생성합니다.

> num_chars -> int, default : 255

생성할 텍스트의 최대 문자 수를 정의합니다.

> gpt_max_audio_tokens -> int, default : 604

오토리그레시브 모델이 처리할 수 있는 최대 Mel 토큰 수를 설정합니다.

> gpt_max_text_tokens -> int, default : 402

오토리그레시브 모델에서 처리 가능한 최대 텍스트 토큰 수를 설정합니다.

> gpt_max_prompt_tokens -> int, default : 70

프롬프트 입력에 사용할 최대 토큰 수를 설정합니다.

> gpt_layers -> int, default : 30

오토리그레시브 모델의 layer 수를 설정합니다.

> gpt_n_model_channels -> int, default : 1024

모델의 채널 수를 정의합니다.

> gpt_n_heads -> int, default : 16

오토리그레시브 모델에서 멀티헤드 어텐션의 헤드 수를 설정합니다.

