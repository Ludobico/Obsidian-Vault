---
Created: 2024-07-29
---

`raise` 는 [[Python]] 에서 **예외를 발생시키는 데 사용되는 키워드**입니다. 예외를 발생시키면 프로그램의 실행이 중단되고, 해당 예외를 처리하기 위한 코드가 실행됩니다. 대표적인 예외처리로는 다음과 같습니다.

### ValueError
- 잘못된 값이 입력된 경우

```python
def check_age(age):
    if age < 0:
        raise ValueError("나이는 음수일 수 없습니다.")
    print(f"입력된 나이는 {age}입니다.")

try:
    check_age(-1)
except ValueError as e:
    print(e)
```

### TypeError
- 잘못된 타입이 사용된 경우

```python
def add_numbers(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("두 인수는 숫자여야 합니다.")
    return a + b

try:
    result = add_numbers(10, '20')
except TypeError as e:
    print(e)

```

### IndexError
- 리스트의 인덱스가 범위를 벗어난 경우

```python
def get_element(lst, index):
    if index >= len(lst):
        raise IndexError("인덱스가 리스트의 범위를 벗어났습니다.")
    return lst[index]

try:
    element = get_element([1, 2, 3], 5)
except IndexError as e:
    print(e)

```

### KeyError
- 딕셔너리에 존재하지 않는 키에 접근하려는 경우

```python
def get_value(dictionary, key):
    if key not in dictionary:
        raise KeyError(f"'{key}' 키가 딕셔너리에 없습니다.")
    return dictionary[key]

try:
    value = get_value({'a': 1, 'b': 2}, 'c')
except KeyError as e:
    print(e)

```

### FileNotFoundError
- 파일을 찾을 수 없는 경우

```python
def read_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"'{filename}' 파일을 찾을 수 없습니다.")

try:
    content = read_file('non_existent_file.txt')
except FileNotFoundError as e:
    print(e)

```

### Custom Error

```python
class CustomError(Exception):
    pass

def do_something():
    raise CustomError("사용자 정의 예외가 발생했습니다.")

try:
    do_something()
except CustomError as e:
    print(e)

```

