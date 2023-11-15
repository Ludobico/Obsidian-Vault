앞에서 <font color="#ffff00">클래스 내에 정의된 함수를 메서드</font> 라고 부른다고 했습니다. 그리고 메서드의 첫 번째 인자는 항상 `self` 여야 한다고 했습니다. 하지만 메서드의 첫 번째 인자가 항상 `self` 여야 한다는 것은 사실 틀린 말입니다. 이번 절에서는 파이썬 클래스에서 `self` 의 정체를 확실히 이해해 봅시다.
먼저 다음과 같은 두 개의 메서드가 정의된 `Foo` 클래스를 만들어 봅시다. 여기서 눈여겨 봐야할 점은 `func1()` 메서드의 첫 번째 인자가 `self` 가 아님에도 클래스를 정의할 때 에러가 발생하지 않는다는 점입니다.

```python
class Foo:
  def func1():
    print("function 1")
  def func2(self):
    print("function 2")

if __name__ == "__main__":
  foo = Foo()
```

일단 클래스를 정의했으니 해당 클래스에 대한 인스턴스를 생성해 보겠습니다. 그리고 생성된 인스턴스를 통해 인스턴스 메서드를 호출해보겠습니다. `Foo` 클래스의 `func2` 메서드는 메서드의 인자가 `self` 뿐이므로 실제 메서드를 호출할 때는 인자를 전달할 필요가 없습니다.

```python
class Foo:
  def func1():
    print("function 1")
  def func2(self):
    print("function 2")

if __name__ == "__main__":
  foo = Foo()
  foo.func2()
```

```
function 2
```

위 코드에서 메서드를 호출한 결과를 보면 화면에 정상적으로 function 2가 출력된 것을 볼 수 있습니다. 참고로 `func2` 메서드의 첫 번째 인자는 `self` 지만 호출할 때는 아무것도 전달하지 않는 이유는 첫 번째 인자인 `self` 에 대한 값은 파이썬이 자동으로 넘겨주기 때문입니다.

그렇다면 `func1` 메서드처럼 메서드를 정의할 때부터 아무 인자도 없는 경우에는 어떻게 될까요? 다음과 같이 인스턴스를 통해 `func1()` 을 호출해 보면 오류가 발생합니다. 오류 메시지를 살펴보면 **func1() 은 인자가 없지만 하나를 받았다.** 라는 것을 볼 수 있습니다. 이는 앞서 설명한 것처럼 파이썬 메서드의 첫 번째 인자로 항상 인스턴스가 전달되기 때문에 발생하는 문제입니다.

```python
class Foo:
  def func1():
    print("function 1")
  def func2(self):
    print("function 2")

if __name__ == "__main__":
  foo = Foo()
  foo.func1()
```

```
    foo.func1()
TypeError: Foo.func1() takes 0 positional arguments but 1 was given
```

이번에는 `self` 의 정체를 좀 더 확실히 밝혀보기 위해 파이썬 내장 함수인 <font color="#ffc000">id</font> 를 이용해 <font color="#ffff00">인스턴스가 메모리에 할당된 주솟값</font>을 확인해보겠습니다. 다음 코드처럼 `Foo` 클래스를 새로 정의합니다. `func2` 메서드가 호출될 때 메서드의 인자로 전달되는 `self` 의 id 값을 화면에 출력하는 기능이 추가되었습니다.

```python
class Foo:
  def func1():
    print("function 1")
    
  def func2(self):
    print(id(self))
    print("function 2")
```

`Foo` 클래스를 새롭게 정의했으므로 인스턴스를 다시 만든 후 `id()` 내장함수를 이용해 인스턴스가 할당된 주소를 확인해 봅시다.

```python
class Foo:
  def func1():
    print("function 1")
  def func2(self):
    print(id(self))
    print("function 2")

if __name__ == "__main__":
  f = Foo()
  print(id(f))
```

```
43219856
```

생성된 인스턴스가 메모리의 <font color="#ffff00">43219856 번지</font> 에 있음을 확인할 수 있습니다. 참고로 이 값은 해당 코드의 실행 환경에 영향을 받게 되므로 여러분이 직접 실행했을 떄 이 값과 다른 값이 나올 수 있습니다.

위 코드에서 `f` 와 생성된 인스턴스의 관계를 그림으로 나타내면 아래 그림과 같습니다. `Foo` 클래스에 대한 인스턴스는 메모리의 43219856 번지부터 할당돼 있고 변수 `f` 는 인스턴스의 주솟값을 담고 있습니다. 일단 인스턴스가 할당된 메모리 주솟값을 기억해두기 바랍니다. 곧 놀라운 광경을 목격할 수 있을 것입니다.

![[Pasted image 20231115155738.png]]

이번에는 인스턴스 `f` 를 이용해 `func2` 메서드를 호출해보기 바랍니다. 다음 코드를 살펴보면 `func2` 메서드를 호출할 때 아무런 값도 전달하지 않았습니다.

```python
class Foo:
  def func1():
    print("function 1")
  def func2(self):
    print(id(self))
    print("function 2")

if __name__ == "__main__":
  f = Foo()
  f.func2()
```

```
43219856
```

실행 결과를 살펴보면 43219856 이라는 값이 출력되는 것을 확인할 수 있습니다. `Foo` 클래스를 저의할 때 `id(self)` 를 출력하게 했는데 `id(self)` 의 값이 바로 43219856 인 것입니다. 이 값은 위 그림에서 볼 수 있듯이 `f` 라는 변수가 바인딩하고 있는 인스턴스의 주솟값과 동일합니다. 즉, <font color="#ffff00">클래스 내에 정의된 self는 클래스 인스턴스</font>임을 알 수 있습니다. 