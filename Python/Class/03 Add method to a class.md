앞서 정의한 `BusinessCard` 클래스는 클래스 내부에 변수나 함수가 없었습니다. 그래서 인스턴스를 만들었음에도 해당 인스턴스로 할 수 있는 일이 별로 없었습니다. 이번에는 `BusinessCard` 클래스에 사용자로부터 데이터를 입력받고 이를 저장하는 기능을 수행하는 함수를 추가해보겠습니다. 참고로 <font color="#ffff00">클래스 내부에 정의돼 있는 함수를 특별이 메서드(method)</font> 라고 합니다. 

다음 코드는 `BusinessCard` 클래스에 `set_info` 라는 메서드를 추가한 것입니다. 메서드를 정의할 때도 함수를 정의할 떄와 마찬가지로 <font color="#ffff00">def</font> 키워드를 사용합니다. `set_info` 메서드는 네 개의 인자를 받는데, 그중 `name` `email` `addr` 은 사용자로부터 입력받은 데이터를 메서드로 전달할 때 사용하는 인자입니다. 그렇다면 메서드의 첫 번째 인자인 `self` 는 무엇일까요?

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.emial = email
    self.addr = addr

if __name__ == "__main__":
  card1 = BusinessCard()
```

파이썬 클래스에서 `self` 의 의미를 정확히 이해하는 것이 중요하지만 아직 제대로 설명하기는 조금 이른 감이 있습니다. 일단 <font color="#ffff00">클래스 내부에 정의된 함수인 메서드의 첫 번째 인자</font>는 반드시 `self` 여야 한다고 외우길 바랍니다.

위 코드에서 메서드 내부를 살펴보면 메서드 인자로 전달된 `name` `email` `addr` 값을 `self.name` `self.email` `self.addr` 이라는 변수에 대입하는 것을 볼 수 있습니다. 앞서 여러 번 설명한 것처럼 파이썬에서 대입은 바인딩을 의미합니다. 따라서 `set_info` 메서드의 동작은 아래 그림과 같이 메서드 인자인 `name` `email` `addr` 이라는 변수가 가리키고 있던 값을 `self.name` `self.email` `self.addr` 이 바인딩하는 것입니다.

![[Pasted image 20231115150315.png]]

`BusinessCard` 클래스를 새롭게 정의했으므로 새롭게 정의된 클래스로 인스턴스를 생성해 봅시다. 붕어빵에 비유해 보면 붕어빵을 굽는 틀을 새롭게 바꿨으므로 새롭게 붕어빵을 구워보는 것입니다.

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.emial = email
    self.addr = addr

if __name__ == "__main__":
  member1 = BusinessCard()
  print(member1)
```

```
<__main__.BusinessCard object at 0x000001953AC6B670>
```

새롭게 정의된 `BusinessCard` 클래스는 `set_info` 라는 메서드를 포함하고 있습니다. 따라서 `member1` 인스턴스는 `set_info` 메서드를 호출할 수 있습니다.

그림을 보면 `member1` 이라는 클래스 인스턴스를 통해 `set_info` 라는 메서드를 호출할 수 있음을 확인할 수 있습니다. 단, 메서드에 인자를 전달하기 위해 괄호를 입력하면 인자가 네 개가 아니라 세 개로 표시됩니다. 앞서 `set_info` 메서드를 정의할 때는 `self` `name` `email` `addr` 의 네 개의 인자가 사용됐는데 메서드를 호출할 때는 왜 세 개만 사용될까요?

![[Pasted image 20231115150649.png]]

일단 다음 코드처럼 파이썬이 알려주는 대로 세 개의 인자만 `set_info` 메서드의 입력으로 전달합니다. 항상 그렇듯이 파이썬에서 에러가 발생하지 않았다면 정상적으로 코드가 실행된 것을 의미합니다.

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.emial = email
    self.addr = addr

if __name__ == "__main__":
  member1 = BusinessCard()
  member1.set_info("Yuna Kim", "yunakim@naver.com", "Seoul")
```

`set_info` 메서드는 메서드 인자로 전달된 값을 `self.name` `self.email` `self.addr` 로 바인딩했습니다. 그런데 현재 사용 가능한 변수는 클래스 인스턴스인 `member1` 뿐입니다. 어떻게 하면 `member1` 을 통해 `self.name` `self.email` `self.addr` 에 접근할 수 있을까요?

`self.name` `self.email` `self.addr` 과 같이 `self.변수명` 과 같은 형태를 띠는 변수를 <font color="#ffff00">인스턴스 변수</font> 라고 합니다. <font color="#ffff00">인</font><font color="#ffff00">스턴스 변수는 클래스 인스턴스 내부의 변수를 의미</font>합니다. 위 코드에서 `member1` 이라는 인스턴스를 생성한 후 `set_info` 메서드를 호출하면 메서드의 인자로 전달된 값을 인스턴스 내부 변수인 `self.name` `self.email` `self.addr` 이 바인딩하는 것입니다. 클래스를 정의하는 순간에는 여러분이 생성할 인스턴스 이름이 `member1` 인지 모르기 때문에 `self` 라는 단어를 대신 사용하는 것입니다.

