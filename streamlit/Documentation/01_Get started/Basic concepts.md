- [[#Streamlit Main Concepts|Streamlit Main Concepts]]
- [[#Basic concepts|Basic concepts]]
	- [[#Basic concepts#1. Stream 앱 실행|1. Stream 앱 실행]]
	- [[#Basic concepts#2. 데이터 흐름과 스크립트 재실행|2. 데이터 흐름과 스크립트 재실행]]
	- [[#Basic concepts#3. Streamlit의 클라이언트-서버 구조|3. Streamlit의 클라이언트-서버 구조]]
	- [[#Basic concepts#4. 빠른 개발 루프|4. 빠른 개발 루프]]
- [[#Key Streamlit Commands|Key Streamlit Commands]]
- [[#Notes|Notes]]

## Streamlit Main Concepts

[[streamlit]] 은 [[Python]] 스크립트를 **인터랙티브한 웹 앱으로 변환**하는 간단하고 강력한 프레임워크입니다. 이 가이드는 streamlit 앱의 구조, 실행 방식, 데이터 흐름 등 기본 개념을 설명합니다.

## Basic concepts

### 1. Stream 앱 실행

Streamlit 앱은 일반 Python 스크립트에 Streamlit 명령어를 추가한 후, `streamlit run` 명령어로 실행합니다.

```bash
streamlit run your_script.py
```

- 실행 시:
	- 로컬 Streamlit 서버가 시작되고, 기본 웹 브라우저에서 앱이 새 탭으로 열립니다.
	- 앱은 캔버스처럼 작동하며, 차트, 텍스트, 위젯, 테이블 등을 그릴 수 있습니다.

- 예시: 간단한 텍스트와 차트 추가

```python
import streamlit as st

st.text("Hello, Streamlit!")
st.line_chart([1, 2, 3, 4, 5])
```

- Streamlit 1.10.0 이상에서는 루트 디렉토리에서 앱을 실행할 수 없습니다. 스크립트를 별도의 디렉토리에 저장하세요.

### 2. 데이터 흐름과 스크립트 재실행

Streamlit은 고유한 데이터 흐름을 통해 앱을 **동적으로 업데이트**합니다. 화면에 변화가 필요할 때마다 Python 스크립트를 처음부터 끝까지 재실행합니다.

- 재실행 트리거
	- 소스 코드 수정 시
	- 사용자가 위젯(슬라이더, 텍스트 입력, 버튼 등)과 상호작용 시
	- 위젯의 `on_change` 또는 `on_click` 콜백 시

- 예시 : 진행 바 업데이트

```python
import streamlit as st
import time

st.write("Starting a long computation...")
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f"Iteration {i+1}")
    bar.progress(i + 1)
    time.sleep(0.1)

st.write("...and now we're done!")
```

![[Pasted image 20250424153358.png]]

### 3. Streamlit의 클라이언트-서버 구조

Streamlit 앱은 클라이언트-서버 구조로 작동합니다.
- 서버 : python 백엔드로, 앱의 연산을 처리합니다. `streamlit run` 을 실행한 기기에서 동작
- 클라이언트 : 사용자가 브라우저로 보는 프론트엔드
- 로컬 개발 시, 동일 기기에서 서버와 클라이언트를 모두 실행, 서버는 한 기기에서 실행되고 클라이언트 네트워크를 통해 연결됩니다.

### 4. 빠른 개발 루프

Streamlit은 코딩과 결과 확인 간의 빠른 피드백 루프를 제공합니다.
- 소스 파일 저장 시 Streamlit이 변경을 감지하고 앱 재실행을 제안
- "Always rerun" 옵션을 선택하면 코드 변경 시 앱이 자동 업데이트
- 편집기와 브라우저를 나란히 배치해 코드와 앱을 동시에 확인하는 것이 좋습니다.

## Key Streamlit Commands

Streamlit은 다양한 명령어를 제공해 앱에 콘텐츠를 추가합니다. 모든 명령어는 API 문서에서 확인 가능합니다.

- `st.text` : 원시 텍스트를 표시

```python
st.text("Raw text example")
```

- `st.write` : 텍스트,차트, 데이터프레임 등 다양한 데이터를 렌더링

```python
st.write("Hello, Streamlit!")
st.write({"data": [1, 2, 3]})
```

- `st.line_chart`, `st.bar_chart` : 간단한 차트 그리기

```python
st.line_chart([10, 20, 30])
```

## Notes

1. 이전 설치 과정에서 `streamlit --version` 오류가 발생했다면, 가상 환경 활성화(`source myenv/bin/activate` 또는 `myenv/Scripts/activate`)를 확인하고, Streamlit을 재설치하세요.

2. Streamlit 기본 개념 문서를 참고해 앱 구조와 동작 원리를 심화 학습하세요.

3. 캐싱 ,세션 상태 등 고급 개념은 [[Advanced concepts]] 에서 확인하세요.

