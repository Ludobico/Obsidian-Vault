typing은 다양한 타입 어노테이션을 위해 사용하는 모듈입니다. 이 모듈은 [[Python]] 3.5 버전 이상부터 사용할 수 있습니다.

## Simple example
---
A 씨는 어느날 다음과 같은 파이썬 코드를 작성했습니다.
```python
def sum_list(numbers: list) -> int:
    return sum(n for n in numbers)


result = sum_list([1, 2, 3, 4])
print(result)
```

A씨가 작성한 `sum_list()` 는 리스트 자료형을 인수로 받아 리스트의 모든 요소의 값을 더하여 반환하는 함수입니다. 이 코드에는 `numbers: list, -> int` 처럼 A씨가 좋아하는 타입 어노테이션을 적용하였습니다. 그리고 A씨는 코드를 테스트해 보고자 다음과 같이 변경해 보았습니다.

```python
def sum_list(numbers: list) -> int:
    return sum(n for n in numbers)


result = sum_list([1, 2, '3', 4])
print(result)
```

`sum_list()` 의 인수로 `[1,2,'3',4]` 처럼 <font color="#ffff00">정수가 아닌 문자열</font>을 요소로 전달했기 때문에 오류가 발생하는 코드입니다. 이렇게 작성하고 다음처럼 mypy 명령을 사용하여 파이썬 타입 체크를 해보았습니다.
```bash
pip install mypy
```

```bash
c:\projects\pylib\ch17>mypy typing_sample.py
Success: no issues found in 1 source file
```

하지만 안타깝게도 mypy는 오류를 감지하지 못합니다. 왜냐하면 `sun_list()`의 매개변수가 `number : list` 처럼 리스트 자료형이기만 하면 타입 체크에 통과하기 때문입니다.

mypy 실행 시 입력 매개변수인 `numbers` 의 모든 요소도 int형 인지를 체크할 수 있도록 하려면 어떻게 프로그램을 수정해야 할까요

다음은 <font color="#00b050">typing</font> 모듈을 사용한 풀이입니다.
```python
from typing import List

def sum_list(numbers: List[int]) -> int:
    return sum(n for n in numbers)

result = sum_list([1, 2, '3', 4])
print(result)
```

typing 모듈을 사용하면 `numbers: list` 대신 `numbers : List[int]` 처럼 사용할 수 있습니다. 여기서는 `numbers : List[int]` 는 <font color="#ffff00">numbers가 리스트 자료형이고 각 요소는 모두 int형이어야 한다는 뜻</font>입니다. 이제 다시 mypy로 타입을 검사해보면
```bash
c:\projects\pylib\ch17>mypy typing_sample.py
typing_sample.py:8: error: List item 2 has incompatible type "str"; expected "int"
Found 1 error in 1 file (checked 1 source file)
```

