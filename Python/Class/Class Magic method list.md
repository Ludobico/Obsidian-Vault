매직 메서드(Magic method) 는 <font color="#ffff00">언더스코어</font>(`__`) <font color="#ffff00">로</font> <font color="#ffff00">시작하고 끝나는 특별한 메서드</font>로, [[Python]] 의 [[Class]] 에서 사용되어 객체의 특정 동작을 지정하거나 커스터마이징하는 데 활용됩니다. 이러한 메서드들은 특정한 상황에서 자동으로 호출되며, 클래스를 사용하는 코드가 특정 연산을 수행할 때 매직 메서드가 실행됩니다. 다양한 매직 메서드를 통해 객체의 라이프 사이클, 연산자 오버로딩, 컨텍스트 관리 등을 제어할 수 있습니다.

- [[#`__init__(self, ...)`|`__init__(self, ...)`]]
- [[#`__str__(self)` , `__repr__(self)`|`__str__(self)` , `__repr__(self)`]]
- [[#`__len__(self)`|`__len__(self)`]]
- [[#`__getitem__(self, key)` `__setitem__(self, key, value)` `__delitem__(self, key)`|`__getitem__(self, key)` `__setitem__(self, key, value)` `__delitem__(self, key)`]]

## `__init__(self, ...)`
---
인스턴스가 생성될 떄 호출되는 생성자 메서드로 초기화 작업을 수행합니다.

```python
class Magic_method:
  def __init__(self, name, age):
    self.age = age
    self.name = name

if __name__ == "__main__":
  a = Magic_method("John", 36)
  print(a.name)
  print(a.age)
```

```
John
36
```

## `__str__(self)` , `__repr__(self)`
---
`str()` 또는 `repr()` 함수를 사용할 때 호출되는 메서드. <font color="#ffff00">객체를 문자열로 표현</font>합니다.

```python
class Magic_method:
  def __init__(self, name, age):
    self.age = age
    self.name = name
  
  def __str__(self) -> str:
    return f"{self.name} is {self.age} years old"

if __name__ == "__main__":
  a = Magic_method("John", 36)
  print(str(a))
```

```
John is 36 years old
```

## `__len__(self)`
---
`len()` 함수를 사용할 때 호출되는 메서드로써 객체의 길이를 반환합니다.

```python
class Magic_method:
  def __init__(self, item):
    self.item = item
  def __len__(self):
    return len(self.item)

if __name__ == "__main__":
 a = Magic_method([1, 2, 3, 4])
 print(len(a))
```

```
4
```

```python
class Magic_method:
  def __init__(self, item):
    self.item = item

if __name__ == "__main__":
 a = Magic_method([1, 2, 3, 4])
 print(len(a))
```

```
    print(len(a))
TypeError: object of type 'Magic_method' has no len()
```

## `__getitem__(self, key)` `__setitem__(self, key, value)` `__delitem__(self, key)`
---
인덱싱 및 슬라이싱을 지원하기 위한 메서드입니다.

```python
class Magic_method:
  def __init__(self, item):
    self.item = item
  
  def __getitem__(self, index):
    return self.item[index]

if __name__ == "__main__":
 a = Magic_method([1, 2, 3, 4])
 print(a[0])
```

```
1
```

