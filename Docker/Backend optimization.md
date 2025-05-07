
## Docker 배포 시

[[Docker]] 를 통해 애플리케이션을 배포했을때 **호스트 머신의 CPU 코어 수는 변경되지 않습니다** Docker 컨테이너는 호스트 머신의 하드웨어 자원을 공유하며, 기본적으로 호스트의 모든 CPU 코어를 사용할 수 있습니다. 하지만 몇 가지 중요한 점을 고려해야합니다.

- Docker는 컨테이너에 별도의 CPU 제한을 설정하지 않으면 호스트 머신의 모든 CPU 코어를 사용할 수 있도록 설정합니다. 즉, 컨테이너 내부에서 실행되는 애플리케이션(unicorn / FastAPI)은 호스트의 코어를 인식하고 활용할 수 있습니다. CPU 코어 수는 아래의 [[Python]] 코드로 확인 가능합니다.

```python
import os

cores = os.cpu_count()

print(cores)
```

- [[Docker compose]] 파일에서 `deploy.resources` 를 통해 리소스를 제한하거나, `--cpus` 또는 `cpuset-cpus` 같은 옵션으로 특정 CPU 코어 사용을 제한할 수 있습니다.
	- 예 : `resources.limits.cpus : 4`


## Uvicorn workers

uvicorn의 workers는 각 워커가 독립적인 Python 프로세스로 실행되며, CPU 코어를 활용해 병렬로 요청을 처리합니다. `workers` 수를 설정할 때 CPU 코어 수를 고려하는 이유는, 각 워커가 CPU 코어를 효율적으로 사용해야 성능이 최적화되기 때문입니다.

### 권장 worker 수

일반적으로 아래로 설정합니다.

- <font color="#ffff00">CPU 코어 수</font>
- <font color="#ffff00">CPU 코어 수 * 2</font>
- <font color="#ffff00">CPU 코어 수 * 2 + 1</font>

`docker-compose.yaml` 의 `command` 에 `--workers` 옵션을 추가하여 워커 수를 설정할 수 있습니다.

```yaml
command: uvicorn main:app --host 0.0.0.0 --port 7076 --workers 24
```

### 주의점

- <font color="#ffff00">메모리 사용량</font> : 워커 수를 늘리면 각 워커가 메모리를 추가로 사용합니다. 워커를 설정하기 전에 컨테이 메모리 제한(`resources.limits.memory`)과 호스트의 메모리 용량을 확인하세요, **이는 AI 모델 사용시 차지하는 VRAM 도 포함**됩니다.
- <font color="#ffff00">GPU 작업</font> : FastAPI가 GPU를 직접적으로 사용하지 않더라도, GPU 작업이 CPU와 메모리를 공유하므로 워커 수를 너무 높게 설정하면 리소스 경합이 발생할 수 있습니다.
- <font color="#ffff00">부하 테스트</font> : `ab` 또는 `locust` 같은 도구로 부하 테스트를 수행해 최적의 워커 수를 찾으세요.

