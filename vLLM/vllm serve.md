`vllm serve` 명령어는 [[vLLM]] 을 **OpenAI API와 호환**되는 HTTP 서버로 변환해주는 핵심 기능입니다. 이를 통해 작성한 모델을 외부 애플리케이션에서 즉시 호출할 수 있습니다.

`vllm serve` 는 내부적으로 FastAPI를 사용하여 서버를 구동하며, vLLM의 핵심 강점인 [[PagedAttention]] 과 [[Continuous Batching]] 을 온라인 환경에서 구축합니다.

## Key Arguments

### Model Setup

`--model` **필수** : 사용할 모델의 이름이나 로컬 경로 (HuggingFace Repo ID 등)

`--tokenizer` : 모델과 다른 토크나이저를 쓸 경우 지정, 기본값은 모델과 동일

`--revision` : 모델의 특정 브랜치나 커밋 해시를 지정

`--dodwnload-dir` : 모델을 다운로드할 특정 디렉토리를 지정


### Hardware & Performance

GPU 메모리 관리와 직결되는 중요한 섹션입니다.

`--tensor-parallel-size` , `-tp` : 모델을 분할하여 실행할 **GPU 개수** 입니다. 예를 들어 Llama-70b를 GPU 4개로 돌릴 시 `4` 로 입력합니다.

`--gpu-memory-utilization` : vLLM이 점유할 **GPU 메모리 비율** 입니다. (기본값 0.9)

`--max-model-len` : 모델이 처리할 최대 문맥 길이([[context length]]) 입니다. 메모리가 부족하면 이값을 줄여야합니다.

`--enforce-eager` : CUDA graph를 사용하지 않고 Eager 모드로 실행합니다. 메모리 절약에 도움을 주나 속도는 약간 느려집니다.

`--kv-cache-dtype` : kv 캐시의 데이터 타입을 정의합니다.

### Serving config

`--host` : 서버 주소 ( 기본값 0.0.0.0, 모든 접근 허용)

`--port` : 서버 포트 ( 기본값 8000 )

`--api-key` : API 호출 시 인증을 위한 키 설정입니다. 보안을 위해 권장됩니다.

`--served-model-name` : API 응답 시 표시될 모델 이름입니다. (기본값 `--model` 과 동일)

## Usage Example

### 기본 실행

```bash
vllm serve "facebook/opt-125m"
```

### GPU 4개를 사용

```bash
vllm serve "meta-llama/Llama-3-70B-Instruct" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --port 8080
```

### 보안 설정이 포함된 실행

```bash
vllm serve "mistralai/Mistral-7B-v0.1" \
    --api-key "my-secret-key-123" \
    --served-model-name "my-chat-model"
```

