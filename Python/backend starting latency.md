---
Created: 2026-08-10T00:00:00.000Z
---
# 백엔드(FastAPI+uvicorn) 시작 latency 측정 방법

## 0. 전체 그림
process 시작 → (import) → FastAPI app 생성 → uvicorn이 lifespan 실행 → `Uvicorn running on ...`
이 중 어디가 느린지 구간별로 쪼개서 재는 방법.

## 1. 구간 ①: "lifespan 전"까지 (import + 모듈 초기화)

`main.py` 맨 위, `time` 이후 최대한 이른 지점에 시작 시각을 기록:

```python
import time
START_TIME = time.perf_counter()

import asyncio
# ... 나머지 import
```

`lifespan` 함수 맨 첫 줄(`try:` 전)에서 여기까지 걸린 시간을 로그:

```python
async def lifespan(app: FastAPI):
    agent_event_logger.info(
        "agent lifespan start",
        context={"elapsed_since_process_start_ms": round((time.perf_counter() - START_TIME) * 1000, 2)},
    )
```

## 2. 구간 ②: lifespan 내부 각 단계

lifespan 안의 무거운 초기화 단계(embedding/qdrant/reranker/llm/graph 등) 앞뒤로 `perf_counter` 체크포인트:

```python
step_start = time.perf_counter()
embedding = get_embedding()
agent_event_logger.info(
    "lifespan step",
    context={"step": "get_embedding", "elapsed_ms": round((time.perf_counter() - step_start) * 1000, 2)},
)
```

같은 패턴을 `get_reranker()`, `get_llm()` x2, `create_graph()`, Langfuse init, feedback DB init 등 각 단계마다 반복.

## 3. 구간 ③: lifespan 완료 후 uvicorn이 실제로 뜨기까지

`Uvicorn running on ...` 로그는 uvicorn 내부(`Server.startup()`)에서 lifespan 완료 **이후** 찍히므로, 코드로 직접 못 잡고 `uvicorn.error` 로거에 필터를 걸어서 가로채야 함:

```python
import logging

class _StartupTimingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage().startswith("Uvicorn running on"):
            elapsed_ms = round((time.perf_counter() - START_TIME) * 1000, 2)
            agent_event_logger.info("agent backend ready", context={"elapsed_ms": elapsed_ms})
        return True  # 원래 로그도 그대로 출력되게

logging.getLogger("uvicorn.error").addFilter(_StartupTimingFilter())
```

→ ①+②+③ 로그를 합치면 "총 시간 = import 비용 + lifespan 작업 비용 + uvicorn 부팅 오버헤드"로 분해됨.

## 4. import 자체를 더 잘게 쪼개서 보기 (`-X importtime`)

어떤 패키지 import가 오래 걸리는지 보려면:

```bash
uv run python -X importtime -m main 2> importtime.log
```

이 로그는 **정렬돼 있지 않고**, import된 순서대로 부모→자식 트리 형태로 찍힘. 컬럼: `self(자기 시간) | cumulative(자식 포함 누적) | 모듈명`, 들여쓰기 = 트리 깊이.

**누적시간 기준 상위 N개 보기**:

```bash
grep "^import time:" importtime.log | sort -t'|' -k2 -n -r | head -30
```
- `-t'|'`: `|` 기준 필드 구분
- `-k2`: 두 번째 필드(cumulative) 기준
- `-n -r`: 숫자 내림차순

⚠️ **주의**: 이렇게 정렬하면 원래의 부모-자식 들여쓰기 구조가 깨짐(값이 비슷한 게 우연히 옆에 붙어 보일 뿐, 진짜 트리 관계 아님). 특정 모듈의 진짜 하위 트리를 보고 싶으면 **정렬 안 한 원본 로그**에서 그 모듈 줄을 찾아 그 아래 더 깊게 들여쓰기된 줄들을 직접 읽어야 함:
```bash
grep -n "모듈이름" importtime.log
```

## 5. 측정 시 흔한 함정

- **`python -m main` + `uvicorn.run("main:app", ...)` (문자열)**: uvicorn이 내부적으로 `main` 모듈을 **다시 import**해서(이미 `__main__`으로 한 번 실행된 상태인데, `sys.modules`엔 `"main"`이라는 키가 없어서 재실행됨), 로그가 값이 다른 2줄로 찍힘. 첫 번째 줄(더 큰 값)이 진짜 "프로세스 시작부터" 총 시간이고, 두 번째 줄은 "재-import 시점부터"라 실제보다 작게 나옴. 헷갈리면 `uvicorn.run("main:app", ...)` → `uvicorn.run(app, ...)`(객체 직접 전달)로 바꾸면 재-import 자체가 안 일어남 (Docker의 `uvicorn main:app ...` CLI 실행에는 영향 없음 — 그건 이 파이썬 코드를 안 거치는 별개 경로).
- **콜드 vs 웜 스타트**: 같은 프로세스를 여러 번 재기동하면 OS 파일 캐시 때문에 두 번째부터 import가 훨씬 빨라짐. "진짜 첫 실행" 기준으로 재야 실제 배포 환경(항상 새 프로세스로 뜨는 콜드 스타트) 수치와 비슷함.
- **provider/기능별 SDK를 함수 최상단에서 항상 import**하면, 실제 안 쓰는 provider의 무거운 의존성(예: `langchain_litellm` → `sentence_transformers` → `torch`)까지 매번 로딩 비용을 냄 → 실제 쓰는 분기 안으로 lazy import 시 그 provider를 안 쓰면 비용 자체가 안 남.