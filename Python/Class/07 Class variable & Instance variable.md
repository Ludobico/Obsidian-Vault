저번 [[06 Class name space]] 에서는 클래스의 네임스페이스와 인스턴스의 네임스페이스, 그리고 그 둘 사이의 관계에 대해 배웠습니다. 이번 절에서는 많이 어려워하는 개념 중 하나인 **클래스 변수(class variable)** 과 **인스턴스 변수(instance variable)** 에 대해 살펴보겠습니다.

다음은 은행 계좌를 클래스로 표현한 것입니다. `Account` 클래스에는 생성자와 생성자 `__init__` 은 클래스의 인스턴스가 생성될 때 자동으로 호출되는 함수라며 소멸자 `__del__` 은 클래스의 인스턴스가 소멸될 때 자동으로 호출되는 함수입니다.

```python
class Account:
  num_accounts = 0
  def __init__(self, name):
    self.name = name
    Account.num_accounts += 1
  
  def __del__(self):
    Account.num_accounts -= 1

```

`Account` 클래스에는 `num_accounts` 와 `self.name` 이라는 두 종류의 변수가 있습니다. `num_accounts` 처럼<font color="#ffff00"> 클래스 내부에 선언된 변수를 클래스 변수</font>라고 하며, `self.name` 과 같이 <font color="#ffff00">self 가 붙어 있는 변수를 인스턴스 변수</font>라고 합니다. <font color="#ffff00">클래스 변수는 Account 클래스의 네임스페이스에 위치</font>하며, `self.name` 과 같은 <font color="#ffff00">인스턴스 변수는 인스턴스의 네임스페이스에 위치</font>하게 됩니다.

그렇다면 언제 클래스 변수를 사용해야 하고 언제 인스턴스 변수를 사용해야 할까요? 이에 대한 답은 간단한 코드를 작성해보면서 천천히 설명해 드리겠습니다. 여러분이 은행에 가서 계좌를 개설하면 새로운 계좌가 하나 개설됩니다. 이러한 상황을 파이썬으로 표현하면 다음과 같이 `Account` 클래스의 인스턴스를 생성하는 것에 해당합니다.

```python
class Account:
  num_account = 0
  def __init__(self, name):
    self.name = name
    Account.num_account += 1
  
  def __del__(self):
    Account.num_account -= 1

if __name__ == "__main__":
  kim = Account("Kim")
  lee = Account("Lee")
```

생성된 `kim` 과 `lee` 인스턴스에 계좌 소유자 정보가 제대로 저장돼 있는지 확인해 봅시다. 각 계좌에 대한 소유자 정보는 인스턴스 변수인 `name` 이 바인딩 하고 있습니다.

```python
class Account:
  num_account = 0
  def __init__(self, name : str) -> str:
    self.name = name
    Account.num_account += 1
  
  def __del__(self):
    Account.num_account -= 1

if __name__ == "__main__":
  kim = Account("Kim")
  lee = Account("Lee")
  print(kim.name)
  print(lee.name)
```

```
Kim
Lee
```

그렇다면 지금까지 은행에서 개설된 총 계좌는 총 몇 개일까요? 정답은 `kim` 과 `lee` 에게 하나씩 개설되었기 때문에 두 개겠죠?
`kim` 인스턴스나 `lee` 인스턴스를 통해 `num_accounts` 라는 이름에 접근하면 총계좌개설개수가 2개로 나오는 것을 알 수 있습니다.

```python
class Account:
  num_account = 0
  def __init__(self, name : str) -> str:
    self.name = name
    Account.num_account += 1
  
  def __del__(self):
    Account.num_account -= 1

if __name__ == "__main__":
  kim = Account("Kim")
  lee = Account("Lee")
  print(kim.num_account)
  print(lee.num_account)
```

```
2
2
```

`kim.num_account` 에서 먼저 인스턴스의 네임스페이스에서 `num_account`를 찾았지만 해당 이름이 없어서 클래스의 네임스페이스로 이동한 후 다시 해당 이름을 찾았고 그 값이 반환된 것임을 아실 것입니다. [[06 Class name space]]

이처럼 여러 <font color="#ffff00">인스턴스 간에 서로 공유해야 하는 값은 클래스 변수를 통해 바인딩</font>해야 합니다. 왜냐하면 파이썬은 인스턴스의 네임스페이스에 없는 이름은 클래스의 네임스페이스에서 찾아보기 때문에 이러한 특성을 이용하면 클래스 변수가 모든 인스턴스에 공유될 수 있기 때문입니다. 참고로 클래스 변수에 접근할 때 아래와 같이 클래스 이름을 사용할 수도 있습니다.

```python
class Account:
  num_account = 0
  def __init__(self, name : str) -> str:
    self.name = name
    Account.num_account += 1
  
  def __del__(self):
    Account.num_account -= 1

if __name__ == "__main__":
  kim = Account("Kim")
  lee = Account("Lee")
  print(Account.num_account)
```

```
2
```

지금까지 작서안 코드에서 클래스 변수와 인스턴스 변수를 그림으로 나타내면 아래 그림과 같습니다. 앞으로 클래스 변수와 인스턴스 변수가 헷갈릴 때마다 이 그림을 기억하기 바랍니다.

![[Pasted image 20231116134056.png]]

