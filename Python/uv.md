---
Created: 2025-04-16
---
## Overview
---

<font color="#ffff00">uv</font> 는 Rust로 작성된 [[Python]] 패키지 매니저로, pip, pip-tools, virualenv를 대체하는 툴입니다. 주요 특징으로는

- 패키지 설치와 의존성 해결이 pip보다 **10~100배 빠름**.
- 전역 캐시로 중복 의존성을 줄여 공간 절약
- pip의 모호한 에러 대신 직관적인 메시지 제공
- 가상 환경 관리, 의존성 잠금, 패키지 설치 등을 단일 CLI 로 처리

## Installation
---

uv는 다음과 같은 커맨드로 실행 가능합니다.

#### windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### pip

```bash
pip install uv
```

설치후 아래의 커맨드로 정상적으로 설치가 완료되었는지 확인할 수 있습니다.

```
uv
```

```
uv --version
```

## Commands
---
### initialize project

```
uv init [프로젝트명] [옵션]
```

새 Python 프로젝트를 생성하고 `pyproject.toml` 을 설정합니다. 기본 디렉토리 구조와 가상환경을 생성합니다.

생성된 `pyproject.toml` 파일은 아래와 같은 구조로 이루어져있습니다.

```toml
[project]
name = "myproject"
version = "0.1.0"
dependencies = []
```

### Install packages

```
uv add [패키지명] [옵션]
```

프로젝트의 `pyproject.toml` 에 의존성을 추가하고, 가상 환경에 즉시 설치하며 yarn 과 비슷한 `uv.lock` 파일을 자동으로 생성합니다.

```bash
uv add requests
```

```bash
uv add "numpy>=1.21.0"  # 특정 버전 지정
```

- `pyproject.toml` 에 추가

```toml
[project]
dependencies = ["requests", "numpy>=1.21.0"]
```

- 대응하는 pip 워크플로우

```bash
pip install requests
```

```bash
pip install "numpy>=1.21.0"
```

### Remove packages

```
uv remove [패키지명]
```

`pyproject.toml` 에서 의존성을 제거하고 가상 환경에서도 삭제합니다.

```bash
uv remove requests
```

- 대응하는 pip 워크플로우

```bash
pip uninstall requests
```

### Synchronize dependencies

```bash
uv sync
```

`pyproject.toml` 과 `uv.lock` 에 따라 가상 환경을 동기화하고 불필요한 패키지는 제거합니다.

### Run script

```bash
uv run python script.py
```

```bash
uv run pytest
```

