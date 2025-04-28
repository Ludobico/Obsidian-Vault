---
Created: 2025-04-16
---
- [[#Overview|Overview]]
- [[#Installation|Installation]]
		- [[#windows|windows]]
		- [[#macOS and Linux|macOS and Linux]]
		- [[#pip|pip]]
- [[#Commands|Commands]]
	- [[#Commands#initialize project|initialize project]]
	- [[#Commands#Create virual environment|Create virual environment]]
	- [[#Commands#Install packages|Install packages]]
	- [[#Commands#Install packages 2|Install packages 2]]
	- [[#Commands#Remove packages|Remove packages]]
	- [[#Commands#Synchronize dependencies|Synchronize dependencies]]
	- [[#Commands#Run script|Run script]]
- [[#Install pytorch on uv|Install pytorch on uv]]
		- [[#Run script#cuda 11.8|cuda 11.8]]
		- [[#Run script#cuda 12.4|cuda 12.4]]
		- [[#Run script#cuda 12.6|cuda 12.6]]


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

새로운 **Python 프로젝트를 초기화**합니다. 프로젝트 구조를 설정하고, 필요한 구성 파일을 생성합니다.

- `pyproject.toml` : 프로젝트 메타데이터와 의존성을 정의하는 설정 파일을 생성합니다.
- `.gitignore` : Python 프로젝트에 적합한 기본 .gitignore 파일을 생성합니다.
- `README.md` : 프로젝트 설명을 위한 기본 파일을 생성합니다.
- `.python-version` : 프로젝트에서 사용할 Python 버전을 지정할 수 있습니다.

가상환경을 **자동으로 생성되지 않지만**, 이후 `uv add` 나 `uv sync` 같은 명령어를 실행하면 `.venv` 디렉토리에 가상환경이 생성됩니다.

생성된 `pyproject.toml` 파일은 아래와 같은 구조로 이루어져있습니다.

```toml
[project]
name = "myproject"
version = "0.1.0"
dependencies = []
```

### Create virual environment

독립적인 **Python 가상환경을 생성**합니다. 프로젝트 초기화 없이 가상환경만 필요할 때 사용됩니다.

- 저장된 디렉토리(기본값 `.venv`)에 가상환경을 생성합니다.
- Python 인터프리터와 기본 패키지만 포함하며, `pyproject.toml` 이나 기타 프로젝트 파일은 생성하지 않습니다.
- `--name` 옵션을 사용하면 `.venv` 대신 원하는 이름의 디렉토리로 가상환경을 생성할 수 있습니다.

```bash
uv venv --name uv_venv
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

### Install packages 2

```
uv pip intall [패키지명] [옵션]
```

`uv pip install` 은 `pip intall` 의 `uv` 버전으로, 가상환경에 패키지를 직접 설치하는 데 초점을 맞춘 명령어입니다. **pyproject.toml 이나 uv.lock 과 직접 연동되지 않으며**, 프로젝트의 의존성 관리보다는 독립적인 패키지 설치에 적합합니다.

### Remove packages

```
uv remove [패키지명]
```

`pyproject.toml` 에서 의존성을 제거하고 가상 환경에서도 삭제합니다.

```bash
uv remove requests
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

## Install pytorch on uv

1. `uv pip install` 은 `uv add` 와 달리 `pyproject.toml` 이나 `uv.lock` 에 의존하지 않고, **직접 지정한 인덱스에서 패키지를 설치**합니다.

#### cuda 11.8

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### cuda 12.4

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### cuda 12.6

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

2. `uv add` 명령어를 통해 `pyproject.toml` 에 패키지를 추가하고, `uv.lock` 파일을 업데이트해 의존성을 관리합니다. 

```bash
uv add torch torchvision torchaudio
```

3. [[Pytorch/CUDA/CUDA|CUDA]] 가 정상적으로 작동하는지 확인합니다.

```python
import torch

def main():
    print(torch.cuda.is_available())


if __name__ == "__main__":
    main()

```

```
True
```
