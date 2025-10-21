---
Created: 2025-04-18
---

## Overview

VSCode (혹은 cursorAI) 에서 [[Python]] 을 사용하면서 , **Run python file(shift + n)** 으로 파일을 실행할떄 모듈을 import 하게되는데, 현재 실행되는 파일을 기준으로 실행하게 됩니다.

```python
# module/a.py 에서 실행
from utils import highlight_print
```

그에 따라 <font color="#ffff00">루트프로젝트를 기준</font>으로 `utils` 폴더에있는 `highlight_print` 함수를 실행하면 다음과 같은 에러가 나오게 됩니다.

```bash
ModuleNotFoundError: No module named 'utils'
```

## How to solve it?

### Regist root project in system path

이를 해결하려면, **모든 파일의 system path에 루트프로젝트를 입력**하거나 **터미널 실행시 루트프로젝트를 등록** 하는 방법으로 이 문제를 해결할 수 있습니다.

파이썬에서 실행할 파일에 아래처럼 현재 폴더를 기준으로 루트 프로젝트를 `os` 모듈로 찾아서 system path로 등록합니다.

```python
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
```

### install -e .

파이썬 프로젝트를 진행할 때, 모든 파일마다 `sys.path.append` 코드를 추가해 프로젝트 루트를 잡아주는것은 번거롭고 비효율적입니다. 이 문제를 근본적으로 해결하는 방법은 프로젝트를 **하나의 설치 가능한 파이썬 패키지**로 만드는 것입니다. 이 방식은 [[uv]] 와 `pip`  모두 호환됩니다.

- 핵심 원리
프로젝트의 구조와 정보를 `pyproject.toml` 이라는 표준 설계도에 정의합니다. `pip` 나 `uv` 같은 도구는 이 설계도를 읽어 프로젝트의 루트가 어디인지, 어떤 폴더가 패키지인지 스스로 파악하게 됩니다.

#### pyproject.toml

프로젝트의 핵심 역할을 하는 설정 파일입니다. 프로젝트 루트 폴더에 `pyproject.toml` 파일을 만들고 아래 내용을 작성합니다.

- **`[project]`**: 프로젝트의 이름과 버전 등 기본 정보를 정의합니다.
    
- **`[tool.setuptools.packages.find]`**: 이 부분이 가장 중요합니다. `setuptools`에게 프로젝트 폴더 내에서 **패키지로 인식할 모든 폴더를 자동으로 찾아달라**고 지시합니다. 이 덕분에 새 폴더를 추가해도 설정을 변경할 필요가 없습니다.

```toml
# pyproject.toml

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my-project"  # 프로젝트 이름으로 변경하세요
version = "0.1.0"

[tool.setuptools.packages.find]
# 비워두면 현재 폴더(루트)를 기준으로 자동으로 모든 패키지를 찾습니다.
```

위 설계도를 바탕으로, 현재 가상 환경에 프로젝트를 <font color="#ffff00">편집 가능 모드</font> 로 설치합니다. 이 과정은 **프로젝트당 한 번**만 하면 됩니다. 이 명령은 실제 파일을 복사하는 대신, 소스 코드를 직접 가리키는 링크를 생성하여 변경이 즉시 반영되게 합니다.

#### uv

```bash
# 프로젝트 루트 폴더에서 실행
uv pip install -e .
```

#### pip

```bash
# 프로젝트 루트 폴더에서 실행
pip install -e .
```

