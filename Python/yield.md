
대부분의 [[Python]] 에서 일반적으로 함수는 어떤 결과 값을 <font color="#ffc000">return</font> 키워드를 이용해서 반환을 하는데요, 하지만 파이썬에서는 함수가 <font color="#ffc000">yield</font> 키워드를 이용해서 다소 다른 방식으로 결과 값을 제공할 수 있습니다.

알파벳 `a` `b` `c` 를 결과 값으로 반환하는 함수를 작성해보겠습니다.

```python
def return_abc():
  return list("ABC")
```

위 함수를 `yield` 키워드를 이용해서 작성해보면

```python
def yield_abc():
  yield "A"
  yield "B"
  yield "C"
```

가장 먼저 눈에 두드러지는 차이는 `return` 키워드를 사용할 때는 결과값을 딱 한 번만 제공하는데, `yield` 키워드는 결과값을 여러 번 나누어서 제공한다는 것입니다.

`for` 루프를 사용해서 위 함수를 호출하여 얻은 결과를 화면에 출력해보겠습니다.

```python
for ch in return_abc():
  print(ch)
```

```bash
A
B
C
```

```python
for ch in yield_abc():
  print(ch)
```

```bash
A
B
C
```

함수를 사용하는 측면에서 보면 두 함수는 큰 차이가 없어보이는데요

<font color="#ffff00">함수를 호출한 결과 값을 바로 출력</font>하여 도대체 각 함수가 정확히 무엇을 반환하는지 알아보겠습니다.

```python
>>> print(return_abc())
['A', 'B', 'C']
>>> print(yield_abc())
<generator object yield_abc at 0x7f4ed03e6040>
```

`return_abc()` <font color="#ffff00">함수는 리스트를 반환</font>하고, `yield_abc()` 함수는 <font color="#ffff00">제너레이터(generator)를 반환</font>한다는 것을 알 수 있습니다.

여기서 우리는 `yield` 키워드를 사용하면 제너레이터를 반환한다는 것을 알 수 있는데요. 과연 generator는 어떤 개념일까요

generator에 대한 타이핑과 설명은 [[typing]] 에서 확인할 수 있습니다.
## 제너레이터(generator)
---
파이썬에서 제너레이터는 쉽게 말해서 <font color="#ffff00">여러 개의 데이터를 미리 만들어 놓지 않고 필요할 때마다 즉석해서 하나씩 만들어낼 수 있는 객체를 의미</font>합니다.

예를 들어, 위에서 작성한 예제 코드를 알파벳 하나를 만드는데 1초가 걸리도록 수정해 보겠습니다.

```python
import time

def return_abc():
  alphabets = []
  for ch in "ABC":
    time.sleep(1)
    alphabets.append(ch)
  return alphabets
```

위 함수를 호출한 결과를 `for` 루프로 돌려보면 3초가 흐른 후에 `a` `b` `c` 가 한 번에 출력이 되는 것을 볼 수 있습니다.

```python
for ch in return_abc():
  print(ch)
```

```bash
# 3초 경과
A
B
C
```

이 번에는 `yield` 키워드를 이용해서 동일한 결과 값을 제공하는 함수를 작성해보겠습니다.
```python
import time

def yield_abc():
  for ch in "ABC":
    time.sleep(1)
    yield ch
```

위 함수를 호출한 결과를 `for` 루프로 돌려보면 1초 후에 `a` 를 출력하고, 또 1초 후에 `b`, 1초 후 `c` 가 출력이 될 것입니다.

```python
for ch in yield_abc():
  print(ch)
```

```bash
# 1초 경과
A
# 1초 경과
B
# 1초 경과
C
```

