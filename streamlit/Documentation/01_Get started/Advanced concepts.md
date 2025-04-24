
[[streamlit]] 의 고급 개념은 앱의 성능 최적화, 사용자 상호작용 관리, 데이터 흐름 제어를 위한 도구를 제공합니다. 이 가이드는 **캐싱, 세션 상태, 데이터 흐름 관리** 등의 주요 고급 기능을 설명합니다.

## Caching

Streamlit은 계산 비용이 높은 작업을 최적화하기 위해 캐싱을 제공합니다. 캐싱은 데이터를 메모리에 저장해 동일한 입력에 대해 함수를 재실행하지 않도록 합니다.

### 1. @st.cache_data

데이터를 반환하는 함수(예 : 데이터 로드, 전처리)에 사용됩니다.

- 예시 : csv 파일 로드

```python
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()
st.write(data)
```

- 특징
	- 입력이 변경되지 않으면 캐시된 결과를 반환
	- 데이터 직렬화/역직렬화를 자동 처리

### 2. @st.cache_resource

전역 리소스(예 : 머신러닝 모델, 데이터베이스 연결)에 사용됩니다.

- 예시 : 머신러닝 모델 로드

```python
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model():
    return RandomForestClassifier()

model = load_model()
st.write("Model loaded!")
```

- 특징
	- 리소스 객체를 메모리에 유지
	- 캐시된 리소스는 앱 재실행 시 재사용.

## Session State

세션 상태는 사용자 상호작용 데이터를 저장해 앱의 상태를 유지합니다. Streamlit은 스크립트 재실행 시 변수가 초기화되므로, 세션 상태를 사용해 값을 보존합니다.

```python
import streamlit as st

if "counter" not in st.session_state:
    st.session_state.counter = 0

def increment():
    st.session_state.counter += 1

st.button("Increment", on_click=increment)
st.write(f"Count: {st.session_state.counter}")
```

- 특징
	- `st.session_state` 는 딕셔너리처럼 동작
	- 위젯 상호작용(예 : 버튼 클릭, 입력) 시 상태 업데이트 가능
	- 동일 브라우저 탭 내에서 데이터 유지

## Data Flow Management

Streamlit의 데이터 흐름은 스크립트가 위에서 아래로 실행되며, 사용자 입력에 따라 재실행됩니다. 이를 효과적으로 관리하는 방법은 다음과 같습니다.

### 1. 위젯 상태 관리

위젯(예 : 슬라이더, 텍스트 입력)은 세션 상태와 연동해 값을 유지합니다.

- 예시 : 슬라이더 값 저장

```python
import streamlit as st

if "slider_value" not in st.session_state:
    st.session_state.slider_value = 50

st.session_state.slider_value = st.slider("Select a value", 0, 100, st.session_state.slider_value)
st.write(f"Selected: {st.session_state.slider_value}")
```

### 2. 콜백 함수

위젯의 `on_change` 또는 `on_click` 을 사용해 특정 동작을 트리거합니다.

- 예시 : 버튼 클릭 시 동작

```python
import streamlit as st

def update_name():
    st.session_state.name = st.session_state.temp_name

st.text_input("Enter name", key="temp_name", on_change=update_name)
st.write(f"Name: {st.session_state.get('name', '')}")
```

### 3. 효율적인 재실행

- 불필요한 연산을 피하려면 `@st.cache_data` 또는 `@st.cache_resource` 를 사용
- 세션 상태를 활용해 중복 입력을 최소화

## Additional Features

### 1. 앱 레이아웃 최적화

Streamlit은 컬럼, 사이드바, 확장 가능한 컨테이너를 제공해 레이아웃을 유연하게 구성합니다.

- 예시 : 컬럼 사용

```python
import streamlit as st

col1, col2 = st.columns(2)
with col1:
    st.write("Column 1 content")
with col2:
    st.write("Column 2 content")
```

### 2. 연결성과 배포

- [Streamlit Community Cloud](https://streamlit.io/cloud) : 앱을 간단히 배포 가능
- 데이터베이스 연결 : 캐싱을 사용해 연결 효율화

```python
import streamlit as st
import sqlite3

@st.cache_resource
def init_db():
    return sqlite3.connect("example.db")
```

