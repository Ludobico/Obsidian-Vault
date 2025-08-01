---
Created: 2025-08-01
---
- [[#Installation|Installation]]
- [[#Examples|Examples]]
	- [[#Examples#1. 리스트에서 하나 선택|1. 리스트에서 하나 선택]]
	- [[#Examples#2. 문자열 입력 받기|2. 문자열 입력 받기]]
	- [[#Examples#3. 다중선택(checkbox)|3. 다중선택(checkbox)]]
	- [[#Examples#4. 비밀번호 입력|4. 비밀번호 입력]]
	- [[#Examples#5. 확인|5. 확인]]


<font color="#ffff00">InquirerPy</font> 는 **터미널 기반 인터렉티브 입력**을 쉽게 만들어주는 [[Python]] 라이브러리입니다. 대표적인 프롬프트 유형은 다음과 같습니다.

| 타입         | 설명                |
| ---------- | ----------------- |
| `input`    | 문자열 입력            |
| `list`     | 리스트 중 하나 선택 (화살표) |
| `checkbox` | 다중 선택             |
| `confirm`  | yes/no            |
| `rawlist`  | 숫자 키로 선택          |
| `password` | 비밀번호 입력 (숨김 처리)   |
| `fuzzy`    | 자동완성 검색 지원 선택     |

## Installation

```bash
pip install InquirerPy
```

## Examples

### 1. 리스트에서 하나 선택

```python
from InquirerPy import prompt

questions = [
    {
        "type": "list",
        "name": "task",
        "message": "작업을 선택하세요:",
        "choices": [
            {"name": "일반 질의응답", "value": 0},
            {"name": "RAG", "value": 1},
            {"name": "Agent", "value": 2},
        ]
    }
]

result = prompt(questions)
print(result["task"])
```

### 2. 문자열 입력 받기

```python
questions = [
    {
        "type": "input",
        "name": "username",
        "message": "사용자 이름을 입력하세요:",
    }
]

```

### 3. 다중선택(checkbox)

```python
questions = [
    {
        "type": "checkbox",
        "name": "features",
        "message": "사용할 기능을 선택하세요:",
        "choices": [
            {"name": "챗봇"},
            {"name": "이미지 생성"},
            {"name": "STT"},
        ]
    }
]
```

### 4. 비밀번호 입력

```python
questions = [
    {
        "type": "password",
        "name": "password",
        "message": "비밀번호를 입력하세요:",
    }
]
```

### 5. 확인

```python
questions = [
    {
        "type": "confirm",
        "name": "continue",
        "message": "계속하시겠습니까?",
        "default": True
    }
]

```

