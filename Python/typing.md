typing은 다양한 타입 어노테이션을 위해 사용하는 모듈입니다. 이 모듈은 [[Python]] 3.5 버전 이상부터 사용할 수 있습니다.

- [[#Simple example (List)|Simple example (List)]]
- [[#Dict|Dict]]
- [[#Tuple|Tuple]]
- [[#set|set]]
	- [[#set#집합 자료형의 특징|집합 자료형의 특징]]
	- [[#set#교집합, 합집합, 차집합 구하기|교집합, 합집합, 차집합 구하기]]
- [[#Generator|Generator]]
- [[#Any|Any]]
- [[#Optional|Optional]]
- [[#Union|Union]]
	- [[#Union#isinstance|isinstance]]
- [[#Callable|Callable]]
- [[#Awaitable|Awaitable]]
- [[#AsyncIterable|AsyncIterable]]
- [[#AsyncIterator|AsyncIterator]]
- [[#AsyncGenerator|AsyncGenerator]]


## Simple example (List)
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

> 파이썬은 3.9 버전 이후부터는 Dict, Tuple, Set 대신 dict, tuple, set 자료형을 그대로 사용할 수 있습니다.

## Dict
---
딕셔너리는 <font color="#00b050">Dict</font> 모듈을 사용합니다.

```python
>>> from typing import Dict
>>> persons: Dict[str, int] = {"홍길동":23, "이한수":34}
>>> persons
{'홍길동': 23, '이한수': 34}

```

## Tuple
---
튜플은 <font color="#00b050">Tuple</font> 을 사용합니다.

```python
>>> from typing import Tuple
>>> hong: Tuple[str, int, bool] = ('홍길동', 23, True)
>>> hong
('홍길동', 23, True)
```

## set
---
집합은 Set 을 사용하여 만들 수 있습니다.

```python
>>> from typing import Set
>>> mark: Set[str] = {"A", "B", "C", "D", "F"}
>>> mark
{'D', 'B', 'C', 'F', 'A'}
```


집합 자료형은 다음과 같이 set 키워드를 사용해 만들 수 있습니다.

```python
>>> s1 = set([1, 2, 3])
>>> s1
{1, 2, 3}
```

위와 같이 `set()` 의 괄호 안에 리스트를 입력하여 만들거나 다음과 같이 문자열을 입력하여 만들 수도 있습니다.

```python
>>> s2 = set("Hello")
>>> s2
{'e', 'H', 'l', 'o'}
```

> 비어 있는 집합 자료형은 s = set() 으로도 만들 수 있습니다.

### 집합 자료형의 특징

그런데 위에서 살펴본 `set("Hello")` 의 결과가 좀 이상합니다. 분명 `Hello` 문자열로 set 자료형을 만들었는데 생성된 자료형에는 l 문자가 하나 빠져 있고 순서도 뒤죽박죽입니다. 그 이유는 `set` 에 다음과 같은 2가지 특징이 있기 때문입니다.

- <font color="#ffff00">중복을 허용하지 않는다</font>
- <font color="#ffff00">순서가 없다(Unordered)</font>

리스트나 튜플은 순서가 있기(ordered) 때문에 인덱싱을 통해 요솟값을 얻을 수 있지만, set 자료형은 순서가 없기(unordered) 때문에 인덱싱을 통해 요솟값을 얻을 수 없습니다.

만약 set 자료형에 저장된 값을 인덱싱으로 접근하려면 다음과 같이 리스트나 튜플로 변환한 후에 해야합니다.

```python
>>> s1 = set([1, 2, 3])
>>> l1 = list(s1)
>>> l1
[1, 2, 3]
>>> l1[0]
1
>>> t1 = tuple(s1)
>>> t1
(1, 2, 3)
>>> t1[0]
1

```

### 교집합, 합집합, 차집합 구하기

set 자료형을 정말 유용하게 사용하는 경우는 <font color="#ffff00">교집합, 합집합, 차집합</font>을 구할 떄입니다.
먼저 다음과 같이 2개의 set 자료형을 만든 후

```python
>>> s1 = set([1, 2, 3, 4, 5, 6])
>>> s2 = set([4, 5, 6, 7, 8, 9])
```

`s1` 과 `s2` 의 <font color="#ffff00">교집합</font>을 구할때는 <font color="#00b050">&</font> 또는<font color="#00b050"> intersection</font> 함수를 사용하여 구할 수 있습니다.

```python
>>> s1 & s2
{4, 5, 6}
```

```python
>>> s1.intersection(s2)
{4, 5, 6}
```

`s1` 과 `s2` 의 <font color="#ffff00">합집합</font> 을 구할때는 <font color="#00b050">|</font> 또는 <font color="#00b050">union</font> 함수를 사용하여 구할 수 있습니다. 이때 중복해서 포함된 값은 1개씩만 표현됩니다.

```python
>>> s1 | s2
{1, 2, 3, 4, 5, 6, 7, 8, 9}
```

```python
>>> s1.union(s2)
{1, 2, 3, 4, 5, 6, 7, 8, 9}
```

<font color="#ffff00">차집합</font>을 구할때는 <font color="#00b050">-</font> 또는 <font color="#00b050">difference</font> 함수를 사용하여 구할 수 있습니다.

```python
>>> s1 - s2
{1, 2, 3}
>>> s2 - s1
{8, 9, 7}
```

```python
>>> s1.difference(s2)
{1, 2, 3}
>>> s2.difference(s1)
{8, 9, 7}
```

## Generator
---

typing 모듈에서 제공하는 `Generator` 는<font color="#ffff00"> 값을 한 번에 하나씩 생성하고 이를 이터레이션(iteration) 할 수 있는 iterator</font> 입니다. 

`Generator` 타입 힌트는 다음과 같은 형태를 가지고 있습니다.

```python
from typing import Generator

def simple_generator() -> Generator[int, str, float]:
    yield 42
    yield "hello"
    return 3.14

gen = simple_generator()

# 이터레이션을 통해 값을 가져옴
value1: int = next(gen)  # 42
value2: str = next(gen)  # "hello"
value3: float = next(gen)  # 3.14
```

```python
from typing import Generator

def simple_generator() -> Generator[int, str, float]:
  yield 42
  yield "Hello"
  yield 3.14

gen = simple_generator()

if __name__ == "__main__":
  for item in gen:
    print(item)
```

> 42
> Hello
> 3.14

위의 코드에서 `Generator[int, str, float]` 는 제너레이터가 `int` 값을 생성하고, 그 다음에는 `str` 값을 생성하며, 마지막으로 `float` 값을 생성하는 것을 나타냅니다.

제너레이터는 `yield` 문을 사용하여 값을 반환하고 호출자에게 제어를 넘기는 특징이 있습니다. 이를 통해 함수가 실행 중에 상태를 유지하면서 값을 계속 생성할 수 있습니다. 이는 <font color="#ffff00">메모리를 효율적으로 사용할 수 있고, 대량의 데이터를 처리할 때 유용</font>합니다.

## Any
---

`Any` 는 <font color="#ffff00">어떤 타입이든 허용되는 동적 타입(dynamic type)을 나타내는 특별한 타입 힌트</font>입니다. 파이썬은 동적 타이핑(dynamic typing) 언어이기 때문에 변수의 타입이 runtime에 결정되기 때문에, `Any` 를 사용하면 어떤 타입이든지 해당 변수나 파라미터에 할당될 수 있다는 것을 나타냅니다.

```python
from typing import Any

def example_function(value : Any) -> Any:
  return value

if __name__ == "__main__":
  result = example_function(42)
  result_str = example_function("hello")
  result_list = example_function([1,2,3])
  print(result)
  print(result_str)
  print(result_list)
```

위의 예제에서 `example_function` 은 어떤 타입의 값을 받든지 상관하지 않고, 동일한 타입의 값을 반환합니다. `Any`를 사용하면 정적 타입 검사는 일부 제약을 갖게 되지만, 동시에 더 유연한 코드 작성이 가능해집니다.

`Any` 를 남용하면 코드의 가독성과 유지보수성이 감소할 수 있으므로, 가능한한 정적 타입 힌트를 사용하여 타입 정보를 명시하는 것이 좋습니다.

## Optional
---
`Optional` 은 <font color="#ffff00">어떤 변수가 특정 타입이거나 None 일 수 있다는 것</font>을 나타냅니다. 이것은 주로 함수의 인자나 반환 값에 사용되며, 해당 변수가 값이 없을 수 있다는 가능성을 표현할 때 유용합니다.

예를 들어, 함수의 파라미터로써 `Optional` 을 사용할 수 있습니다.

```python
from typing import Optional

def greet(name : Optional[str] = None) -> str:
  if name is None:
    return "Hello, Stranger"
  else:
    return "Hello, {0}".format(name)
```

위 함수는 `name` 에 아무값도 전달하지 않으면 `None` 값을 전달하도록 하였습니다.

## Union
---
`Union` 은 <font color="#ffff00">여러 타입 중 하나일 수 있는 경우</font>를 나타냅니다. 일반적으로 파이썬에서는 여러 타입을 허용하는 경우를 다룰 때 `Union` 을 사용합니다.

```python
from typing import Union

def square_or_cube(value : Union[int, float]) -> Union[int, float]:
  if isinstance(value, int):
    return value ** 2
  elif isinstance(value, float):
    return value ** 3
  else:
    raise ValueError("Unsupported type")

if __name__ == "__main__":
  print(square_or_cube(2))
  print(square_or_cube(2.5))
  print(square_or_cube("error"))
```

>4
  15.625
>ValueError: Unsupported type

위의 예제에서 `Union[int, float]` 은 함수 `square_or_cube` 의 파라미터 `value` 와 반환값이 정수 또는 부동 소수점 숫자 중 하나일 수 있음을 나타냅니다. 

### isinstance

`isinstance` 는 파이썬 내장 함수 중 하나로, <font color="#ffff00">주어진 객체가 특정 클래스 또는 특정 타입의 인스턴스인지를 확인하는데 사용</font>됩니다. 

`isinstance` 함수의 일반적인 사용법은 다음과 같습니다.

```python
result = isinstance(object, classinfo)
```

- object : 타입을 확인하고자 하는 객체입니다.
- classinfo : 확인하고자 하는 타입, 클래스, 또는 튜플입니다.

## Callable
---

`Callble` 은 <font color="#ffff00">함수나 메서드의 시그니처(signature)를 표현하는 타입 힌트</font>입니다. 함수가 어떤 파라미터를 받고 어떤 타입의 값을 반환하는지 명시적으로 나타낼 수 있습니다.

`Callable` 은 다음과 같은 형태를 가집니다.

```python
from typing import Callable

def example_function(x : int, y : str) -> float:
  return float(x)

callable_example : Callable[[int, str], float] = example_function

if __name__ == "__main__":
  result = callable_example(1, "Hello")
  print(result)
```
> 1.0

위의 예제에서 `Callable[[int, str], float]` 는 함수가 `int` 타입과 `float` 타입의 파라미터를 받아들이고 `float` 타입의 값을 반환함을 나타냅니다. 이러한 타입 힌트를 사용하면 함수의 인터페이스를 명확하게 정의하고, 코드를 더욱 가독성 있게 만들 수 있습니다.

`Callable` 은 파라미터의 개수와 타입, 그리고 반환 타입을 정의할 수 있습니다. 예를 들어, 다음은 <font color="#ffff00">파라미터가 없는 함수를 나타내는 사용 예제</font>입니다.

```python
from typing import Callable

def no_argument_function() -> None:
    print("No arguments")

callable_no_argument: Callable[[], None] = no_argument_function

callable_no_argument()  # 타입 검사를 통과
```


## Awaitable
---
`Awaitable` 은 <font color="#ffff00">비동기 코드에서 사용되는 타입 힌트 중 하나</font>입니다. 비동기 코드에서는 <font color="#00b050">async</font> 키워드가 사용되며, <font color="#00b050">await</font> 키워드를 사용하여 비동기 작업의 완료를 기다립니다. <font color="#00b050">Awaitable</font> 은 이러한 비동기 함수나 객체를 나타냅니다.

간단한 예제를 통해 설명해보겠습니다. 아래 코드에서 `async def` 로 정의된 `async_finction` 함수는 `Awaitable` 을 사용합니다. 

```python
from typing import Awaitable

async def async_function() -> Awaitable[str]:
    result = await some_async_operation()
    return result
```

여기서 `Awaitable[str]` 은 <font color="#ffff00">비동기적으로 실행되는 함수나 객체가 반환할 것으로 예상되는 타입</font>을 나타냅니다. 위의 예제에서는 문자열(str) 을 기대하고 있습니다.

비동기 함수에서 `Awaitable` 을 사용하지 않아도 일반적으로 에러가 발생하지 않습니다. Python은 동적 타이핑 언어이므로 많은 경우에는 자동으로 타입을 추론할 수 있습니다.

```python
async def async_function() -> str:
    result = await some_async_operation()
    return result
```

## AsyncIterable
---
`AsyncIterable` 은 <font color="#ffff00">비동기적으로 반복 가능한(iterable) 객체를 나타내는 타입</font>입니다. 이는 비동기적으로 값을 생성하는 컨테이너를 나타냅니다. 이러한 객체는 <font color="#00b050">async for</font> 루프에서 사용할 수 있습니다.

간단한 예제로 설명해보겠습니다. 아래는 `AsyncIterable` 을 사용하여 간단한 비동기적으로 값을 생성하는 함수입니다.

```python
from typing import AsyncIterable, Awaitable, List
import asyncio

async def counter(numbers = List[int]) -> Awaitable[AsyncIterable[int]]:
  for number in numbers:
    print(number)

if __name__ == "__main__":
  example_numbers = [0,1,2,3,4,5]
  asyncio.run(counter(example_numbers))
```

>0
  1
  2
  3
  4
  5

## AsyncIterator
---


## AsyncGenerator
---

`AsyncGenerator` 는 typing 모듈에서 제공하는 하나의 타입으로, 비동기 제너레이터(Asynchronous Generator)를 나타냅니다.

비동기 제너레이터는 일반적인 제너레이터와 유사하지만 <font color="#00b050">async def</font> 키워드를 사용하여 정의되며 비동기적으로 동작합니다. 비동기 제너레이터는 <font color="#00b050">async for</font> 루프를 통해 비동기적으로 값을 생성하고 소비됩니다.

`AsyncGenerator` 는 다음과 같이 사용될 수 있습니다.

```python
from typing import AsyncGenerator
import asyncio

async def async_generator_example() -> AsyncGenerator[int, str]:
  for i in range(5):
    yield i
    await asyncio.sleep(0.5)
  yield "done"

async def consume_async_generator():
  async for i in async_generator_example():
    print(i)
    print(type(i))

if __name__ == "__main__":
  asyncio.run(consume_async_generator())
```

>0
  <class 'int'>
  1
  <class 'int'>
  2
  <class 'int'>
  3
  <class 'int'>
  4
  <class 'int'>
  done
  <class 'str'>

위의 코드에서 `AsyncGenerator[int, str]` 는 비동기 제너레이터가 생성하는 값의 타입이 `int` 이고, 완료되면 `str` 을 반환함을 나타냅니다. 이러한 타입 힌팅을 통해 코드를 작성할 때 타입 검사를 할 수 있으며, 가독성을 향상시킬 수 있습니다.


