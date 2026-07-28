---
aliases:
  - vLLM API 엔드포인트
  - vLLM 서빙 API
tags:
  - vllm/api
  - vllm/serving
  - api/reference
  - machine-learning/llm
date_created: 2026-07-28
---

# vLLM 온라인 서빙 엔드포인트 참조 가이드

vLLM은 다양한 인기 AI/LLM 인터페이스(OpenAI, Anthropic, Cohere 등)와 호환되는 HTTP 서버를 제공하며, 자체 관리, 프로파일링 및 개발용 API도 지원합니다.

---

## 📌 목차 (빠른 이동)

- [[#1. OpenAI 호환 서버|1. OpenAI 호환 서버]]
- [[#2. Anthropic API|2. Anthropic API]]
- [[#3. Cohere API|3. Cohere API]]
- [[#4. 풀링(Pooling) API|4. 풀링(Pooling) API]]
- [[#5. 음성 인식(Speech to Text) API|5. 음성 인식(Speech to Text) API]]
- [[#6. 커스텀(Custom) API|6. 커스텀(Custom) API]]
- [[#7. 모니터링 및 기본 API|7. 모니터링 및 기본 API (기본 & 메트릭)]]
- [[#8. 오프라인 API 문서 활성화|8. 오프라인 API 문서 활성화]]
- [[#9. LoRA 동적 로딩|9. LoRA 동적 로딩]]
- [[#10. 프로파일링 API|10. 프로파일링 API]]
- [[#11. SageMaker API|11. SageMaker API]]
- [[#12. 토큰 입력/출력 전용 API|12. 토큰 입력/출력 전용 API]]
- [[#13. 렌더러 & 디렌더러 API|13. 렌더러 & 디렌더러 API]]
- [[#14. 토큰화(Tokenize) API|14. 토큰화(Tokenize) API]]
- [[#15. 탄력적 전문가 병렬 처리(EEP)|15. 탄력적 전문가 병렬 처리(EEP)]]
- [[#16. 개발 모드 전용 API|16. 개발 모드 전용 API (캐시, 가중치 전송, 슬립 모드, Collective RPC)]]
- [[#17. 챗 템플릿(Chat Template) 설정|17. 챗 템플릿(Chat Template) 설정]]
- [[#18. Ray Serve LLM|18. Ray Serve LLM]]

---

## 1. OpenAI 호환 서버
vLLM은 텍스트 생성, 임베딩 및 음성 모델을 위한 표준 OpenAI API 엔드포인트를 지원합니다.

| 엔드포인트 | 메서드 | 적용 가능 모델 | 상세 설명 |
| :--- | :--- | :--- | :--- |
| `/v1/completions` | `POST` | 텍스트 생성 모델 | 기본 텍스트 완성 API. `suffix` 매개변수는 지원되지 않습니다. |
| `/v1/chat/completions` | `POST` | 챗 생성 모델 | 챗 완료 API. `user` 매개변수는 무시됩니다. |
| `/v1/chat/completions/batch`| `POST` | 챗 생성 모델 | 일괄(Batch) 처리용 챗 완료 엔드포인트입니다. |
| `/v1/responses` | `GET` | 텍스트 생성 모델 | 생성된 응답을 조회합니다. |
| `/v1/responses/{id}` | `GET` | 텍스트 생성 모델 | 특정 ID의 응답을 조회합니다. |
| `/v1/responses/{id}/cancel` | `POST` | 텍스트 생성 모델 | 생성 진행 중인 응답을 취소합니다. |
| `/v1/embeddings` | `POST` | 임베딩 모델 | 벡터 임베딩을 생성합니다. |
| `/v1/audio/transcriptions` | `POST` | ASR 모델 | 음성 파일을 텍스트로 변환(전사)합니다. |
| `/v1/audio/translations` | `POST` | ASR 모델 | 음성 파일을 번역합니다. |

> [!NOTE] parallel_tool_calls 작동 방식
> `parallel_tool_calls` 매개변수를 `false`로 설정하면 vLLM은 요청당 0개 또는 1개의 도구 호출(tool call)만 반환합니다. `true`(기본값)로 설정하면 여러 도구 호출을 동시에 반환할 수 있으나, 이는 모델의 동시 호출 설계 여부에 의존합니다.

---

## 2. Anthropic API
Anthropic의 Messages API 규격과 호환되는 엔드포인트를 제공합니다.

- `/v1/messages` (`POST`) — Anthropic Messages API.
- `/v1/messages/count_tokens` (`POST`) — 입력 페이로드의 토큰 수를 계산합니다.

---

## 3. Cohere API
Cohere SDK 및 추론 아키텍처와 호환되는 엔드포인트입니다.

- `/v2/embed` (`POST`) — Cohere Embed API. 멀티모달 모델을 포함한 모든 임베딩 모델과 호환됩니다.
- `/rerank`, `/v1/rerank`, `/v2/rerank` (`POST`) — Cohere Rerank API. Jina AI의 v1 rerank API 규격을 구현하였으며, Cohere의 v1 및 v2 rerank API와 호환됩니다.

---

## 4. 풀링(Pooling) API
특정 풀링 모델용 엔드포인트입니다.

- **분류(Classification) 관련:**
  - `/classify` (`POST`) — 분류 전용 모델에만 적용됩니다.
- **임베딩(Embedding) 관련:**
  - `/v2/embed` (`POST`) — Cohere Embed API 호환.
  - `/v1/embeddings` (`POST`) — OpenAI 호환 Embeddings API.
- **점수 측정(Scoring) 관련:**
  - `/score`, `/v1/score` (`POST`) — 점수 측정 API.
  - `/rerank`, `/v1/rerank`, `/v2/rerank` (`POST`) — Cohere Rerank API.
  - *적용 범위:* 점수 모델 (교차 인코더, 쌍방 인코더, Late-interaction).
- **공통 풀링:**
  - `/pooling` (`POST`) — 모든 풀링 모델에 적용 가능합니다.

---

## 5. 음성 인식(Speech to Text) API
자동 음성 인식(ASR) 및 음성 번역 모델용 연동 엔드포인트입니다.

- `/v1/audio/transcriptions` (`POST`) — 음성 전사(Transcription) API.
- `/v1/audio/translations` (`POST`) — 음성 번역(Translation) API.
- `/v1/realtime` (`WebSocket`/`HTTP`) — 실시간 음성 스트리밍 API.

---

## 6. 커스텀(Custom) API
vLLM 내부 서버 고유 기능에 맞춘 자체 커스텀 엔드포인트입니다.

- `/classify` (`POST`) — 분류 API.
- `/score`, `/v1/score` (`POST`) — 점수 측정 API (교차 인코더, 쌍방 인코더, Late-interaction 모델용).
- `/pooling` (`POST`) — 풀링 API (모든 풀링 모델 대상).
- `/generative_scoring` (`POST`) — 생성 기반 점수 측정 API (생성 태스크를 수행하는 CausalLM 모델 대상). 지정된 `label_token_ids`에 대한 다음 토큰 확률을 계산합니다.

---

## 7. 모니터링 및 기본 API

### 기본 API
서버 상태 확인 및 모델 검색에 사용되는 기본 엔드포인트들입니다.
- `/version` (`GET`) — 현재 구동 중인 vLLM 서버의 버전을 조회합니다.
- `/load` (`GET`) — 현재 서버의 로드 메트릭을 가져옵니다.
- `/v1/models` (`GET`) — 로드되어 사용 가능한 모델 목록을 반환합니다.
- `/health` (`GET`) — 서버 헬스체크 엔드포인트입니다.

### 메트릭 API
- `/metrics` (`GET`) — 프로메테우스(Prometheus) 호환 포맷의 HTTP 메트릭 엔드포인트입니다.

---

## 8. 오프라인 API 문서 활성화
FastAPI의 `/docs` 대화형 UI는 기본적으로 CDN을 통해 스크립트를 로드하므로 인터넷 연결이 필요합니다. 폐쇄망(Air-gapped) 환경에서 오프라인으로 접속할 수 있게 하려면 서버 실행 시 `--enable-offline-docs` 플래그를 지정하십시오.

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --enable-offline-docs
```

---

## 9. LoRA 동적 로딩
서버 실행 중에 LoRA 어댑터를 동적으로 로딩하거나 언로딩하여 모델 체크포인트를 바꿀 수 있는 엔드포인트입니다.

> [!WARNING] 로컬 개발 전용
> LoRA 동적 로딩/언로딩 엔드포인트는 **반드시** 로컬 개발 목적으로만 사용해야 합니다!

- `/v1/load_lora_adapter` (`POST`) — LoRA 어댑터를 동적으로 로드합니다.
- `/v1/unload_lora_adapter` (`POST`) — 로드된 LoRA 어댑터를 언로드합니다.

---

## 10. 프로파일링 API
PyTorch 프로파일러를 실행하여 서버 동작 트레이스(Trace)를 기록합니다.

- `/start_profile` (`POST`) — PyTorch 프로파일러 작동을 시작합니다.
- `/stop_profile` (`POST`) — PyTorch 프로파일러 작동을 정지합니다.

---

## 11. SageMaker API
AWS SageMaker 서빙 컨테이너 규격과 호환되는 엔드포인트입니다.

- `/ping` (`GET`) — 컨테이너 헬스체크용 엔드포인트입니다.
- `/invocations` (`POST`) — `/v1` 엔드포인트와 동일한 내부 추론 함수로 라우팅합니다.

---

## 12. 토큰 입력/출력 전용 API
텍스트가 아닌 토큰 아이디 기준의 엄격한 추론 및 세션 관리를 수행하는 API입니다.

- `/inference/v1/generate` (`POST`) — 원시 토큰 입력을 통한 완성본 생성을 수행합니다.
- `/abort_requests` (`POST`) — 실행 중인 요청을 중단합니다. (서버 시작 시 `--tokens-only` 플래그가 설정된 경우에만 사용 가능)

---

## 13. 렌더러 & 디렌더러 API

### 렌더러 API
- `/v1/completions/render` (`POST`) — 완성 요청 프롬프트를 렌더링합니다.
- `/v1/chat/completions/render` (`POST`) — 챗 완료 요청 프롬프트를 렌더링합니다.

### 디렌더러 API
- `/v1/chat/completions/derender` (`POST`) — 챗 완료 요청을 디렌더링(프롬프트 해제)합니다.
- `/v1/completions/derender` (`POST`) — 완성 요청을 디렌더링합니다.

---

## 14. Tokenize API
서버 내 토크나이저를 직접 디버깅하고 조회하는 엔드포인트입니다.

- `/tokenize` (`POST`) — 일반 텍스트를 토큰 ID 목록으로 변환합니다.
- `/detokenize` (`POST`) — 토큰 ID 목록을 다시 일반 텍스트로 복원합니다.
- `/tokenizer_info` (`GET`) — 로드된 토크나이저의 상세 설정 정보(기본 Jinja 챗 템플릿 포함)를 가져옵니다.

---

## 15. 탄력적 전문가 병렬 처리(EEP)
Elastic Expert Parallelism 구조에서 동적으로 전문가 라우팅 용량을 설정하고 모니터링하는 API입니다.

- `/scale_elastic_ep` (`POST`) — 전문가 확장/축소(Scaling) 연산을 트리거합니다.
- `/is_scaling_elastic_ep` (`GET`) — 현재 전문가 확장이 진행 중인지 여부를 확인합니다.

---

## 16. 개발 모드 전용 API
서버 기동 시 환경 변수로 `VLLM_SERVER_DEV_MODE=1`을 선언하면 잠금 해제되는 엔드포인트들입니다.

> [!CAUTION] 보안 경고
> 이 엔드포인트들을 **절대** 프로덕션 환경에 노출하지 마십시오! 모델 가중치의 직접 변조, 내부 캐시 강제 삭제, 스케줄러 일시 정지 등의 강력한 연산이 가능하여 서비스 거부(DoS)나 보안 취약점을 유발할 수 있습니다.

### 캐시 관리 (Cache Management)
- `/reset_prefix_cache` (`POST`) — 프리픽스 캐시를 완전히 리셋합니다 (진행 중인 서비스에 장애를 초래할 수 있음).
- `/reset_mm_cache` (`POST`) — 멀티모달 캐시를 리셋합니다.
- `/reset_encoder_cache` (`POST`) — 인코더 캐시를 리셋합니다.

### 가중치 전송 및 RLHF (Weight Transfer)
- `/pause` (`POST`) — 추론 엔진 스케줄러를 일시 중지합니다 (서비스 거부 발생).
- `/resume` (`POST`) — 일시 중지된 추론 엔진 스케줄러를 재개합니다.
- `/is_paused` (`GET`) — 스케줄러가 현재 정지 상태인지 조회합니다.
- `/abort_requests` (`POST`) — 스케줄러를 일시 정지하지 않고 모든 진행 중인 요청 혹은 특정 `request_ids`들을 즉시 중단합니다.
- `/init_weight_transfer_engine` (`POST`) — RLHF 훈련을 위한 가중치 전송 엔진을 초기화합니다.
- `/start_weight_update` (`POST`) — 가중치 업데이트 작업을 위해 추론 엔진을 준비 상태로 만듭니다.
- `/update_weights` (`POST`) — 가중치를 동적으로 업데이트하여 모델 동작을 변경합니다.
- `/finish_weight_update` (`POST`) — 가중치 업데이트 절차를 마감합니다.
- `/get_world_size` (`GET`) — 분산 처리 상의 월드 사이즈(World Size)를 조회합니다.

### Collective RPC & 서버 정보
- `/collective_rpc` (`POST`) — 추론 엔진에서 임의의 RPC 메서드를 즉시 실행합니다 (매우 위험).
- `/server_info` (`GET`) — 상세한 서버 구성 설정을 가져옵니다.

### 슬립 모드 (Sleep Mode)
- `/sleep` (`POST`) — 추론 엔진을 대기(Sleep) 모드로 전환하고 GPU 메모리를 일부 반환합니다 (서비스가 중단됩니다).
- `/wake_up` (`POST`) — 대기 상태의 엔진을 깨워 정상 구동합니다.
- `/is_sleeping` (`GET`) — 현재 엔진이 대기 모드인지 조회합니다.

---

## 17. 챗 템플릿(Chat Template) 설정
vLLM에서 챗 완성 규격을 올바르게 동작시키려면 토크나이저 설정 파일 내에 Jinja2 형식의 챗 템플릿이 반드시 존재해야 합니다.

### 수동으로 챗 템플릿 지정하기
적절한 챗 템플릿이 없는 모델인 경우 `--chat-template` 인자값을 사용하여 템플릿 파일 경로를 지정할 수 있습니다.
```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

### 멀티모달 챗 요청 예시 (JSON)
멀티모달 모델을 사용하는 경우, 프롬프트 입력 형식을 아래와 같은 구조화된 배열 형식으로 호출합니다.
```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 감정을 분류해 줘: vLLM은 정말 훌륭합니다!"},
            ],
        },
    ],
)
```

vLLM은 서버 실행 시 챗 템플릿 형식을 자동 감지하여 로그(`"Detected the chat template content format to be..."`)에 남깁니다.
- `string`: 일반 단순 텍스트 프롬프트 (`"Hello world"`).
- `openai`: OpenAI 규격의 딕셔너리 리스트 (`[{"type": "text", "text": "Hello world!"}]`).

이를 수동 강제하려면 서버 시작 시 `--chat-template-content-format` 플래그를 설정하여 오버라이드할 수 있습니다.

---

## 18. Ray Serve LLM
Ray Serve LLM을 사용하여 vLLM 서빙 엔진을 여러 노드로 구성된 대규모 분산 클러스터 상에서 수평 확장할 수 있습니다.
- 오토스케일링, 로드 밸런싱, Back-pressure 제어 기본 제공.
- OpenAI 호환 HTTP API 및 자체 Python API 제공.
- 1개의 GPU에서 수십 대의 멀티 노드 GPU 클러스터로 원활한 코드 변경 없이 확장 가능.
- 대표 연동 예제 소스: [ray_serve_deepseek.py](file:///C:/Users/kcj215/Desktop/repo/Obsidian-Vault/vLLM/examples/ray_serving/ray_serve_deepseek.py)
