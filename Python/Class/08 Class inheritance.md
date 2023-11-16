상속이란 사람이 사망함에 따라 사망자의 재산 및 신분상의 지위에 대한 포괄적인 승계를 의미합니다. 드라마에서 보면 부모님으로부터 많은 재산을 상속받은 사람들을 종종 볼 수 있지요? 상속하는 사람 입장은 잘 모르지만 상속자들은 분명 상속받지 않은 경우보다 대부분 좋은 경우가 많을 것입니다. 특히 많은 재산을 상속받는 경우라면 더욱 좋겠지요?

프로그래밍 책에서 갑자기 상속 이야기를 한 이유는 객체지향 프로그래밍을 지원하는 프로그래밍 언어는 클래스에서 상속 기능을 지원하기 때문입니다. 자식이 부모님으로부터 재산 등을 상속받는 것처럼 다른 클래스에 이미 구현된 메서드나 속성을 상속한 클래스에서는 그러한 메서드나 속성을 그대로 사용할 수 있게 됩니다.

클래스의 상속을 또 다른 관점에서 생각해보면 클래스를 상속한다는 것은 부모 클래스의 능력을 그대로 전달받는 것을 의미합니다. 인간으로 치면 부모로부터 유전형질을 물려받아 부모의 능력을 그대로 물려받는 것과 비슷합니다.

일단 노래를 잘 부르는 부모 클래스가 있다고 생각해 봅시다. 이를 파이썬으로 표현하면 다음과 같이 노래를 부르는 메서드가 포함된 클래스를 정의할 수 있습니다.

```python
class Parent:
  def can_sing(self):
    print("sing a song")
```

`Parent` 클래스를 정의했으니 클래스의 인스턴스를 생성해 보겠습니다. 그리고 노래를 정말 잘 할 수 있는지 메서드를 호출해 확인해보겠습니다.

```python
class Parent:
  def can_sing(self) -> str:
    print("sing a song")

if __name__ == "__main__":
  father = Parent()
  father.can_sing()
```

```
sing a song
```

이번에는 노래를 잘 부르는 `Parent` 클래스로부터 상속받은 운이 좋은 자식 클래스를 정의해 봅시다. 클래스의 이름은 `LuckyChild` 라고 하겠습니다. 클래스를 정의할 떄 다른 클래스로부터 상속받고자 한다면 새로 정의한 클래스 이름 다음에 괄호를 사용해 상속받고자 하는 클래스의 이름을 지정하면 됩니다. `LuckyChild` 클래스는 상속받기 전까지는 아무런 능력이 없어서 내부에 메서드를 구현하지 않고 `pass` 만 적어 주었습니다.

```python
class Parent:
  def can_sing(self) -> str:
    print("sing a song")

class LuckyChild(Parent):
  pass

if __name__ == "__main__":
  child1 = LuckyChild()
  child1.can_sing()
```

```
sing a song
```

이번에는 부모로부터 어떤 능력이나 재산도 상속받지 못한 운이 좋지 않은 자식 클래스 `UnLuckyChild` 를 만들어 보겠습니다. 평범한 서민 클래스라고 볼 수 있겠습니다.

```python
class UnLuckyChild:
  pass
```

운이 좋지 않은 `UnLuckyChild` 클래스에 대한 인스턴스를 생성한 후 노래를 시켜봅시다. 역시나 상속도 받지 못하고 자신도 아무런 메서드가 없으므로 노래를 부를 수 없습니다. 이처럼 능력이나 재산을 상속받지 못했다면 자신이 직접 노래를 잘할 수 있도록 연습해야겠지요? 프로그래밍 측면에서 보면 직접 메서드를 구현해야 한다는 뜻입니다.

```python
class UnLuckyChild:
  pass

if __name__ == "__main__":
  child2 = UnLuckyChild()
  child2.can_sing()
```

```
    child2.can_sing()
AttributeError: 'UnLuckyChild' object has no attribute 'can_sing'
```

인간 세계에서는 상속을 받으면 좋겠구나, 라고 생각되겠지만 프로그래밍할 떄 왜 상속을 해야 하는지 여전히 이해되지 않는 분도 계실 겁니다. 객체지향 프로그래밍에서는 <font color="#ffff00">어떤 클래스를 상속하면 부모 클래스의 모든 것을 내 것처럼 사용할 수 있고, 자신의 클래스에 메서드를 더 구현한다면 플러스알파 가 되는 것</font>입니다. 즉, 부모 클래스의 밑바탕으로 깔고 거기서부터 한 번 더 업그레이드되는 것입니다.

그럼 이번에는 노래도 잘 부르고 춤도 잘 추는 운이 좋은 자식 클래스2 `LuckyChild2` 를 정의해 보겠습니다.

```python
class Parent:
  def can_sing(self) -> str:
    print("sing a song")

class LuckyChild2(Parent):
  def can_dance(self):
    print("Shuffle Dance")
```

`LuckyChild2` 클래스에 대한 인스턴스를 생성한 후 노래와 춤을 시켜봅시다. `LuckyChild2` 클래스는 부모로부터 노래 부르는 능력을 상속받았고 자기 자신이 춤추는 능력을 갖추고 있어 노래도 부르고 춤도 출수 있게 됐습니다.

```python
class Parent:
  def can_sing(self) -> str:
    print("sing a song")

class LuckyChild2(Parent):
  def can_dance(self):
    print("Shuffle Dance")

if __name__ == "__main__":
  child2 = LuckyChild2()
  child2.can_sing()
  child2.can_dance()
```

```
sing a song
Shuffle Dance
```

물론 굳이 상속이라는 기능을 사용하지 않고도 부모 클래스에 구현된 메서드를 그대로 복사해서 새로 정의할 클래스를 코드에 붙여넣는 식으로 사용할 수도 있습니다. 단, 이렇게 하면 같은 기능을 하는 코드가 중복되기 때문에 코드를 관리하기가 어렵고 복사 및 붙여넣기를 해야 하므로 불편합니다. 이에 반해 클래스의 상속이라는 기능을 이용하면 최소한의 코드로도 부모 클래스에 구현된 메서드를 손쉽게 바로 이용할 수 있습니다.

