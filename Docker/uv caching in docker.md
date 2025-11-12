- [[#basic setting|basic setting]]
- [[#파이썬 런타임 캐시 추가 (optional)|파이썬 런타임 캐시 추가 (optional)]]
- [[#Environments|Environments]]


[[uv]] 는 패키지 설치시 <font color="#ffff00">/root/.cache/uv</font> 경로를 사용하지만 [[Docker]] 빌드 레이어는 pip와 다르게 **이 디렉토리를 기본적으로 캐시하지 않습니다.** 그래서 빌드할때마다 매번 패키지를 새로 다운로드하게 됩니다.

이를 해결하기 위해서는 `--mount=type=cache` 옵션으로 uv 캐시 디렉토리를 명시적으로 마운트해야 합니다.

## basic setting

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./

# 링크 경고 방지를 위해 환경변수 설정
ENV UV_LINK_MODE=copy

# uv 캐시를 빌드 캐시로 유지
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install uv && \
    uv sync --frozen --no-dev
```

- `--mount=type=cache, target=/root/.cache/uv`
Docker 빌드 간 uv 캐시를 유지해 다음 빌드부터 빠르게 설치

- `ENV UV_LINK_MODE=copy`
캐시와 work directory 가 다른 파일 시스템일 때 발생하는 hard link 경고 방지

- `uv sync --frozen --no-dev`
lock 파일 그대로 설치하며, dev 의존성은 제외

## 파이썬 런타임 캐시 추가 (optional)

`uv` 는 자체적으로 [[Python]] 런타임도 관리할 수 있습니다.
이 경우 런타임 캐시를 따로 지정하면 불필요한 재설치를 방지할 수 있습니다.

```dockerfile
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install
```

## Environments

| 변수명                 | 역할            | 설명                        |
| ------------------- | ------------- | ------------------------- |
| UV_LINK_MODE=copy   | 하드링크 경고 방지    | 캐시와 프로젝트가 다른 파일 시스템일 때 필요 |
| UV_NO_CACHE=1       | 캐시 비활성화       | 이미지 크기를 최소화하려는 경우 사용      |
| UV_PYTHON_CACHE_DIR | 파이썬 런타임 캐시 경로 | uv python install 시 캐시 유지 |

