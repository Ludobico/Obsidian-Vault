- [[#File Uploader|File Uploader]]
- [[#Theming|Theming]]
- [[#Multi-page Apps|Multi-page Apps]]


[[streamlit]] 은 앱 개발을 더욱 풍부하게 만드는 추가 기능을 제공합니다. 이 가이드는 **파일 업로드, 테마 설정, 페이지 구성 등** streamlit의 유용한 기능을 설명합니다.

## File Uploader

파일 업로더를 사용하면 사용자가 파일(이미지, csv, 텍스트 등)을 업로드해 앱에서 처리할 수 있습니다.

- 사용 예시 : CSV 파일 업로드 및 표시

```python
import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data)
```

- 특징:
	- `type` 파라미터로 허용 파일 형식을 제한합니다.
	- 업로드된 파일은 Python은 `BytesIO` 객체로 처리
	- 다중 파일 업로드 지원 (`accept_multiple_files=True`)

## Theming

Streamlit은 앱의 시각적 스타일을 커스터마이징할 수 있는 테마 기능을 제공합니다.

![[change_theme.gif]]

![[edit_theme.gif]]



- 기본 테마 : Streamlit은 라이트/다크 모드를 제공하며, 시스템 설정에 따라 자동 전환됩니다.
- 커스텀 테마 : `.streamlit/config.toml` 파일을 통해 색상, 글꼴 등을 사용자가 정의합니다.

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

- 적용 방법:
	1. 프로젝트 루트에 `.streamlit/` 폴더 생성
	2. `config.toml` 파일에 테마 추가
	3. 앱 재실행

## Multi-page Apps

Streamlit은 st.Page와 st.navigation을 사용해 다중 페이지 앱을 구성할 수 있습니다. 이를 통해 복잡한 앱을 체계적으로 관리할 수 있습니다.

- 설정 방법:
    
    1. 메인 스크립트(예: streamlit_app.py)에서 페이지 정의 및 탐색 설정.
        
    2. 각 페이지를 별도의 .py 파일로 작성.
        
    3. st.navigation으로 페이지 간 이동 관리.
        
- 예시: 3페이지 앱
    
    - 디렉토리 구조:
        
        ```
        your_app/
        ├── streamlit_app.py
        ├── main_page.py
        ├── page_2.py
        ├── page_3.py
        ```
        
    - **streamlit_app.py**:
        
        ```python
        import streamlit as st
        
        # 페이지 정의
        main_page = st.Page("main_page.py", title="Main Page", icon="🎈")
        page_2 = st.Page("page_2.py", title="Page 2", icon="❄️")
        page_3 = st.Page("page_3.py", title="Page 3", icon="🎉")
        
        # 탐색 설정
        pg = st.navigation([main_page, page_2, page_3])
        
        # 선택된 페이지 실행
        pg.run()
        ```
        
    - **main_page.py**:
        
        ```python
        import streamlit as st
        
        # 메인 페이지 콘텐츠
        st.markdown("# Main Page 🎈")
        st.sidebar.markdown("# Main Page 🎈")
        ```
        
    - **page_2.py**:
        
        ```python
        import streamlit as st
        
        st.markdown("# Page 2 ❄️")
        st.sidebar.markdown("# Page 2 ❄️")
        ```
        
    - **page_3.py**:
        
        ```python
        import streamlit as st
        
        st.markdown("# Page 3 🎉")
        st.sidebar.markdown("# Page 3 🎉")
        ```

![[mpa-v2-main-concepts.gif]]

