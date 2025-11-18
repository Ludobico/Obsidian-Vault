- [[#Overview|Overview]]
- [[#Structure & Schema|Structure & Schema]]
	- [[#Structure & Schema#\[build-system\]|\[build-system\]]]
	- [[#Structure & Schema#\[project\]|\[project\]]]
	- [[#Structure & Schema#\[tool.\*\]|\[tool.\*\]]]

## Overview

<font color="#ffff00">pyproject.toml</font> 은 [[Python]] 프로젝트의 빌드 시스템 요구사항, 프로젝트 메타데이터, 그리고 의존성을 정의하기 위해 고안된 Configuration file 입니다. TOML(Tom's obvious, Minimal Language) 포맷을 사용하여 파편화된 기존 Python 패키징 생태계를 통합하는 표준으로 기능합니다.

## Structure & Schema

pyproject.toml은 크게 세 가지 주요 섹션으로 구분됩니다.

### \[build-system\]

프로젝트를 빌드하기 위해 필요한 도구와 방식을 정의하는 필수 섹션입니다. 소스 코드를 배포 가능한 아티팩트(wheel)로 변환하는 <font color="#ffff00">빌드 백엔드</font>를 지정합니다. 

- main key
	- requires : 필요 패키지 목록
	- build-backend : 빌드 진입점

```toml
# 예시: uv가 기본적으로 사용하는 hatchling 빌드 시스템 설정
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```


### \[project\]

패키지의 이름, 버전, 의존성 등 프로젝트의 정체성을 정의하는 표준화된 섹션입니다. 기존 `setup.py` 의 역할을 대체합니다.

```toml
# 예시: RAG 서비스 프로젝트 메타데이터
[project]
name = "heritage-rag-service"
version = "0.1.0"
description = "RAG 서비스"
readme = "README.md"
requires-python = ">=3.12"
authors = [
    { name = "Developer", email = "dev@example.com" }
]

# 런타임 의존성 (실행에 반드시 필요한 라이브러리)
dependencies = [
    "langchain>=0.3.0",
    "pymupdf>=1.24.0",
    "openai>=1.50.0",
]
```

### \[tool.\*\]

프로젝트에 사용하는 개발 도구들의 설정을 관리하는 섹션입니다. 각 도구별로 존재하던 개별 설정 파일(`.flake8`, `pytest.ini` 등)을 제거하고 단일 파일로 통합합니다.

- `[tool.<도구이름>]` 형식을 따릅니다.

```toml
# 예시 1: uv 패키지 매니저 설정 (개발용 의존성 관리)
[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "ruff>=0.6.0",  # 린터 및 포매터
]

# 예시 2: Ruff (Linter) 설정
[tool.ruff]
line-length = 88
target-version = "py312"

# 예시 3: Pytest (테스트 프레임워크) 설정
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

