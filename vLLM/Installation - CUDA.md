## vLLM GPU Installation Guide

설치 전 시스템 환경이 다음 조건을 충족하는지 확인하세요.
- OS : Linux (Ubuntu 추천)
- [[Python]] : 3.9 ~ 3.12
- GPU
	- NVIDIA : 7.0 이상 (V100, T4, RTX 20 시리즈, A100, H100 등)
	- AMD : ROCm 지원 GPU
	- Intel : GPU 지원
- CUDA : 12.1 이상 권장 (NVIDIA Blackwell의 경우 12.8 이상)

## Pre-built Wheels (권장)

가장 빠르고 안정적인 방법은 [[uv]] 를 통해 설치하는 방법입니다.

`uv` 는 설치 환경에 맞춰 적절한 [[Pytorch]] 인덱스를 자동으로 설치합니다.

### uv

```bash
# 최신 버전 설치
uv pip install vllm --torch-backend=auto
```

### pip

사용중인 CUDA 버전에 맞춰 설치하려면 `--extra-index-url` 을 사용합니다.

```bash
# CUDA 12.1 예시
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

## Nightly 버전 설치

공식 릴리즈 전의 최신 기능이나 버그 수정이 필요할 경우 사용합니다.

```bash
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

## Build from Source

코드 수정이나 특정 환경 최적화가 필요할 경우 사용합니다.

### python 전용 빌드

C++나 커널 코드를 수정하지 않고 Python 코드만 수정할 때 사용합니다.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

### 전체 빌드
C++ 또는 CUDA 커널을 수정해야 할 때 사용합니다.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -e .
```

## Docker 설치

격리 환경이 필요할 경우 vLLM에서 제공하는 [[Docker]] 공식 이미지를 사용해서 설치할 수 있습니다.

```docker
# 공식 이미지 실행 예시
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --network host \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model <모델_이름>
```

