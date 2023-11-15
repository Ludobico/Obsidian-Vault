[[Python]] 의 [[Class]] 에 대해 배웠습니다. 지금까지 배운 내용을 정리해보면 다음과 같스빈다.

- 파이썬의 클래스를 이용하면 프로그래머가 원하는 새로운 타입을 만들 수 있다.
- 생성된 타입은 데이터와 데이터를 처리하는 메서드(함수)로 구성돼 있다.

그럼 지난 시간에 만든 클래스를 다시 한 번 살펴볼까요?

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.email = email
    self.addr = addr
  
  def print_info(self):
    print("--------------------")
    print("Name: ", self.name)
    print("E-mail: ", self.email)
    print("Address: ", self.addr)
    print("--------------------")
```

파이썬의 class 키워드를 통해 `BusinessCard` 라는 새로운 타입을 만들었으니 인스턴스를 생성해야겠죠? 다시 한번 말씀드리면 클래스 인스턴스를 생성하려면 `클래스 이름()` 과 같이 적으면 됩니다. 그리고 생성된 인스턴스에 점(.) 을 찍어서 인스턴스 변수나 메서드에 접근할 수 있었습니다.

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.email = email
    self.addr = addr
  
  def print_info(self):
    print("--------------------")
    print("Name: ", self.name)
    print("E-mail: ", self.email)
    print("Address: ", self.addr)
    print("--------------------")

if __name__ == "__main__":
  member1 = BusinessCard()
  member1.set_info("홍길동", "hong@hg.com", "Seoul")
```

위 코드를 살펴보면 먼저 인스턴스를 생성하고 생성된 인스턴스에 데이터를 입력하는 순으로 코드로 구성돼 있습니다. 붕어빵에 비유해 보면 붕어빵 틀(클래스)을 이용해 팥소를 넣지 않은 상태로 붕어빵을 구운 후 (인스턴스 생성) 나중에 다시 붕어빵 안으로 팥소를 넣는 것과 비슷합니다. 어떻게 하면 클래스 인스턴스 생성과 초깃값 입력을 한 번에 처리할 수 있을까요?

파이썬 클래스에는 <font color="#ffff00">인스턴스 생성과 동시에 자동으로 호출되는 메서드인 생성자가 존재</font>합니다. 참고로 생성자는 C++나 자바같은 객체지향 프로그래밍 언어에도 있는 개념입니다. `__init__(self)` 와 같은 이름의 메서드를 생성자라고 하며, 파이썬에서 `__` 로 시작하는 함수는 모두 특별한 메서드를 의미합니다.

다음은 생성자인 `__init__(self)` 메서드를 가진 MyClass 클래스를 정의한 것입니다. 앞서 설명한 것처럼 <font color="#ffff00">생성자의 첫 번쨰 인자도 항상 self </font>여야합니다. 생성자 내부에는 `print` 문을 통해 간단한 메시지를 출력했습니다.

```python
class MyClass:
  def __init__(self):
    print("객체 생성 완료")

if __name__ == "__main__":
  inst1 = MyClass()
```

```
객체 생성 완료
```

클래스 생성자를 이해했다면 `BusinessCard` 클래스를 수정해 인스턴스의 생성과 동시에 명함에 필요한 정보를 입력받도록 클래스를 새롭게 정의해 봅시다.

```python
class BusinessCard:
  def __init__(self, name, email, addr):
          self.name = name
          self.email = email
          self.addr = addr
  def print_info(self):
          print("--------------------")
          print("Name: ", self.name)
          print("E-mail: ", self.email)
          print("Address: ", self.addr)
          print("--------------------")
```

새로 정의된 `BusinessCard` 클래스의 생성자는 인자가 4개임을 확인할 수 있습니다. 물론 첫 번쨰 인자인 `self` 는 생성되는 인스턴스를 의미하고 자동으로 값이 전달되므로 인스턴스를 생성할 때 명시적으로 인자를 전달해야 하는 것은 3개입니다. 따라서 인스턴스를 생성할 때 3개의 인자를 전달하지 않으면 오류가 발생합니다. 생성자 호출 단계에서 오류가 발생하면 인스턴스도 정상적으로 생성되지 않게 됩니다.

```python
class BusinessCard:
  def __init__(self, name, email, addr):
          self.name = name
          self.email = email
          self.addr = addr
  def print_info(self):
          print("--------------------")
          print("Name: ", self.name)
          print("E-mail: ", self.email)
          print("Address: ", self.addr)
          print("--------------------")

if __name__ == "__main__":
  member1 = BusinessCard()
```

```
    member1 = BusinessCard()
TypeError: BusinessCard.__init__() missing 3 required positional arguments: 'name', 'email', and 'addr'
```

새로 정의된 `BusinessCard` 클래스는 생성자에서 3개의 인자(name, email, addr) 를 받기 때문에 다음과 같이 인스턴스를 생성할 때 3개의 인자를 전달해야 정상적으로 인스턴스가 생성됩니다. `member1` 이라는 인스턴스가 생성된 후에도 인스턴스 메서드를 호출해 인스턴스 변수의 값을 화면에 출력할 수 있습니다. 어떤가요? 클래스의 생성자를 사용하니 인스턴스의 생성과 초깃값 저장을 한 번에 할 수 있어 편리하지요?

```python
class BusinessCard:
  def __init__(self, name, email, addr):
          self.name = name
          self.email = email
          self.addr = addr
  def print_info(self):
          print("--------------------")
          print("Name: ", self.name)
          print("E-mail: ", self.email)
          print("Address: ", self.addr)
          print("--------------------")

if __name__ == "__main__":
  member1 = BusinessCard("Kangsan Lee", "kangsan.lee", "USA")
  member1.print_info()
```

```
--------------------
Name:  Kangsan Lee
E-mail:  kangsan.lee
Address:  USA
--------------------
```

