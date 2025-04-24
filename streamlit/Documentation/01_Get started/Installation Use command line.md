
- [[#Prerequisites|Prerequisites]]
- [[#Installation Steps|Installation Steps]]
	- [[#Installation Steps#1. 가상 환경 생성|1. 가상 환경 생성]]
	- [[#Installation Steps#2. 가상 환경 활성화|2. 가상 환경 활성화]]
		- [[#2. 가상 환경 활성화#for Windows|for Windows]]
		- [[#2. 가상 환경 활성화#for MacOS/Linux|for MacOS/Linux]]
	- [[#Installation Steps#3. Streamlit 설치|3. Streamlit 설치]]
	- [[#Installation Steps#4. 설치 확인|4. 설치 확인]]
	- [[#Installation Steps#5. 간단한 "Hello world" 앱 실행|5. 간단한 "Hello world" 앱 실행]]

# Streamlit Installation (Command Line)

이 가이드는 `venv` 나 `pip` 를 사용해 [[streamlit]] 을 설치하고, 가상 환경을 설정하며, 간단하게 <font color="#ffff00">"Hello World"</font> 앱을 실행하는 방법을 설명합니다. GUI로 [[Python]] 환경을 관리하고 싶다면 Anaconda로 Streamlit 설치를 참고하세요.

## Prerequisites

Streamlit 을 설치하기 위해 다음 조건을 충족해야 합니다.

- [[Python]]:  버전 3.9 >=
- Python environment manager : `venv`
- Python package manager : `pip`
- Mac os 전용 : Streamlit 의존성 설치를 위해 Xcode가 필요합니다.

```
xcode-select --install
```

- IDE : VS Code를 권장

## Installation Steps

### 1. 가상 환경 생성

`venv` 를 사용해 프로젝트별 가상 환경을 만듭니다.

```bash
python -m venv myenv
```

### 2. 가상 환경 활성화

가상 환경을 활성화해 프로젝트에서 격리된 Python 환경을 만듭니다.

#### for Windows

```bash
myenv\Scripts\activate
```

#### for MacOS/Linux

```bash
source myenv/bin/activate
```

활성화 후 터미널에 (myvenv) 접두어가 표시됩니다.

### 3. Streamlit 설치

pip를 사용해 라이브러리를 설치합니다.

```bash
pip install streamlit
```

### 4. 설치 확인

Streamlit이 올바르게 설치되었는지 확인합니다.

```bash
streamlit --version
```

### 5. 간단한 "Hello world" 앱 실행

간단한 앱을 만들어 실행해봅니다.

1. `hello.py` 파일을 생성하고 다음 코드를 추가

```python
import streamlit as st

st.write("Hello, World!")
```

2. 앱 실행

```bash
streamlit run hello.py
```

3. 브라우저가 자동으로 열리며<font color="#ffff00"> http://localhost:8501</font> 에서 앱이 표시됩니다.

