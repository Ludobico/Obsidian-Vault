
도커는 기본설정으로 **컨테이너 내부에서 GPU를 인식하지 못하는 상태**입니다. 이에따라 [[Docker compose]] 에서 따로 추가적인 설정이 필요합니다.


## How to solve it

1. Docker 컨테이너 실행 시 `--gpus all` 옵션을 추가합니다.

```bash
docker run --gpus all
```

2. `docker-compose.yaml` 에 `runtime : nvidia` 추가

아래처럼 `deploy.resources.reservations.devices` 를 설정하면 GPU를 사용하도록 설정할 수 있습니다.

```yaml
version: "3.8"

services:
  gpu_app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gpu_container
    runtime: nvidia  # (구버전) 필요할 수도 있음
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python3 check_gpu.py
```

