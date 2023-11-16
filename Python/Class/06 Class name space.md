클래스 와 인스턴스의 차이를 정확히 이해하는 것은 매우 중요합니다. 이를 위해서는 먼저 **네임스페이스** 라는 개념을 알아야합니다. <font color="#ffff00">네임스페이스라는 것은 변수가 객체를 바인딩할 때 그 둘 사이의 관계를 저장하고 있는 공간</font> 을 의미합니다. 예를 들어 `a = 2` 라고 했을 때 `a` 라는 변수가 `2` 라는 객체가 저장된 주소를 가지고 있는데 그러한 연결 관계가 저장된 공간이 바로 네임스페이스입니다.

[[Python]] 의 [[Class]] 는 새로운 타입(객체)을 정의하기 위해 사용되며, 모듈과 마찬가지로 하나의 네임스페이스를 가집니다. 먼저 `Stock` 클래스를 정의해 봅시다.

```python
class Stock:
  market = "kospi"
```

파이썬에서 `dir` 내장 함수를 호출해보면 리스트로 된 반환값을 확인할 수 있습니다. 여기서 두 개의 언더바로 시작하는 것은 파이썬에서 이미 사용 중인 특별한 것들입니다. 이를 제외하고 보면 조금 전에 정의했던 `Stock` 클래스의 이름이 포함된 것을 확인할 수 있습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  print(dir())
```

```
['Stock', '__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']
```

`dir` 내장함수의 결괏값에 `Stock` 클래스가 들어있기 때문에 앞으로는 프롬프트에 `Stock` 을 입력해도 오류가 발생하기 않습니다. 그러나 `Stock1` 이라는 이름은 존재하지 않기때문에 이를 입력하면 오류가 발생합니다.

```
Stock Stock1 Traceback (most recent call last): File "", line 1, in Stock1 NameError: name 'Stock1' is not defined
```

파이썬에서 클래스가 정의되면 아래 그림과 같이 하나의 독립적인 네임스페이스가 생성됩니다. 그리고 클래스 내에 정의된 변수나 메서드는 해당 네임스페이스 안에 파이썬 딕셔너리 타입으로 저장됩니다. 아래 그림과 같이 `Stock` 이라는 네임스페이스 안에 `"market" : "kospi"` 라는 값을 가진 딕셔너리를 포함합니다.

![[Pasted image 20231116125038.png]]

`Stock` 클래스의 네임스페이스를 파이썬 코드로 확인하려면 클래스의 `__dict__` 속성을 확인하면 됩니다. 딕셔너리 타입에 `"market" : "kospi"` 라는 키와 값 쌍이 존재하는 것을 확인할 수 있습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  print(Stock.__dict__)
```

```
{'__module__': '__main__', 'market': 'kospi', '__dict__': <attribute '__dict__' of 'Stock' objects>, '__weakref__': <attribute '__weakref__' of 'Stock' objects>, '__doc__': None}
```

클래스가 독립적인 네임스페이스를 가지고 클래스 내의 변수나 메서드를 네임스페이스에 저장하고 있으므로 다음과 같이 클래스내의 변수에 접근할 수 있습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  print(Stock.market)
```

```
kospi
```

이번에는 인스턴스를 생성해보겠습니다. 다음과 같이 서로 다른 두 개의 인스턴스를 생성해보기 바랍니다. 생성된 인스턴스에 대한 id 값을 확인해보면 두 인스턴스가 서로 다른 메모리에 위치한 것을 확인할 수 있습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()

  print(id(s1))
  print(id(s2))
```

```
2544717182576
2544717182528
```

파이썬은 인스턴스를 생성하면 인스턴스별로 별도의 네임스페이스를 유지합니다. 즉, 위의 코드를 그림으로 표현하면 아래와 같습니다.

![[Pasted image 20231116125524.png]]

먼저 생성된 `s1` `s2` 인스턴스가 네임스페이스에 있는지 코드를 통해 확인해 봅시다. `dir` 내장함수의 반환값을 확인하면 `s1` `s2` 가 `Stock` 과 마찬가지로 존재하는 것을 확인할 수 있습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()
  print(dir())
```

```
['Stock', '__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 's1', 's2']
```

생성된 `s1` `s2` 인스턴스 각각에 대한 네임스페이스도 확인해봅시다. 클래스 또는 인스턴스에 대한 네임스페이스를 확인하려면 `__dict__` 속성을 확인하면 됩니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()
  print(s1.__dict__)
  print(s2.__dict__)
```

```
{}
{}
```

위 코드를 보면 `s1` 과 `s2` 인스턴스의 네임스페이스는 현재 비어있음을 확인할 수 있습니다. `s1` 인스턴스에 `market` 이라는 변수를 추가해 봅시다. 그런 다음 다시 `__dict__` 속성을 확인해 보면 `"market" : "kosdaq"` 라는 키:값 쌍이 추가된 것을 볼 수 있습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()
  s1.market = "kosdaq"
  print(s1.__dict__)
```

```
{'market': 'kosdaq'}
```

그러나 여전히 `s2` 인스턴스의 네임스페이스는 비어 있는 상태입니다. 현재 `Stock` 클래스와 `s1` `s2` 인스턴스의 네임스페이스를 그림으로 나타내면 아래 그림과 같습니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()
  s1.market = "kosdaq"
  print(s2.__dict__)
```

```
{}
```

![[Pasted image 20231116130157.png]]

현재 `Stock` 클래스로부터 `s1` `s2` 라는 두 개의 인스턴스를 생성했습니다. `s1` 인스턴스는 `market` 이라는 변수를 가지고 있지만 `s2` 의 네임스페이스에는 변수나 메서드가 존재하지 않습니다. 만약 `s1.market` 과 `s2.market` 과 같이 인스턴스를 통해 `market` 이라는 값에 접근하면 어떻게 될까요?

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()
  s1.market = "kosdaq"
  print(s1.market)
  print(s2.market)
```

```
kosdaq
kospi
```

위 코드를 참조하면 `s1.market` 에는 "kosdaq" 입니다. 이것은 위 그림과 같이 `s1` 인스턴스의 네임스페이스에 `"market" : "kosdaq"` 이라는 키:값 쌍이 존재하기 때문에 가능합니다. 그런데 `s2` 인스턴스의 반환값을 살펴보면 조금 이상합니다. 왜냐하면 현재 `s2` 의 네임스페이스(딕셔너리)에는 아무런 값도 존재하지 않기 때문입니다. `s2` 의 네임스페이스에는 변수나 메서드가 존재하지 않지만 `s2.market` 의 값으로 "kospi" 라는 문자열이 반환되는 이유는 아래 그림과 같이 동작하기 때문입니다.

![[Pasted image 20231116130557.png]]

`s2` 인스턴스를 통해 변수에 접근하면 파이썬은 먼저 `s2` 인스턴스의 네임스페이스에서 해당 변수가 존재하는지 찾습니다. `s2` 의 네임스페이스에 해당 변수가 존재하지 않으면 `s2` 인스턴스의 클래스의 네임스페이스로 가서 다시 변수를 찾게 됩니다. 즉, `s2.market` 이라는 문장이 실행되면 `Stock` 클래스의 네임스페이스에 있는 `"market" : "kospi"` 키:값 쌍에서 "kospi" 라는 문자열을 출력하게 됩니다.

이번에는 인스턴스의 네임스페이스에도 없고 클래스의 네임스페이스에도 없는 변수에 접근해 봅시다. 이 경우 `volume` 이라는 값이 `s2` 인스턴스의 네임스페이스에 없으므로 `Stock` 클래스에서 찾게 되는데, `Stock` 클래스의 네임스페이스에도 `volume` 이라는 값이 없으므로 오류가 발생합니다.

```python
class Stock:
  market = "kospi"

if __name__ == "__main__":
  s1 = Stock()
  s2 = Stock()
  s1.market = "kosdaq"
  print(s2.volume)
```

```
    print(s2.volume)
AttributeError: 'Stock' object has no attribute 'volume'
```

