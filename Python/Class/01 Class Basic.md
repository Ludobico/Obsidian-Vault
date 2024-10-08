지금까지 작성한 프로그램을 생각해보면 보통 어떤 데이터가 존재하고 데이터를 조작하는 함수를 통해 데이터를 목적에 맞게 변경한 후 이를 다시 출력하는 형태였습니다. 앞서 살펴본 명함 출력 프로그램([[Class]]) 도 사용자로부터 개인정보를 입력받은 후 해당 데이터를 조작하고 이를 출력하는 함수로 구성돼 있었습니다.

명함 출력에 사용된 이름, 이메일, 주소 등을 효과적으로 관리하기 위해 보통 아래 그림과 같이 각 <font color="#ffff00">데이터에 대해 별도의 리스트 자료구조를 사용</font>합니다. 아래 그림과 같은 자료구조에서 첫 번째 회원에 대한 개인 정보를 얻기 위해서는 `name_list[0]`, `email_list[0]` , `address_list[0]` 과 같이 각 리스트별로 인덱싱을 수행해야 합니다. 참고로 아래그림과 같이 <font color="#ffff00">데이터와 데이터를 처리하는 함수가 분리돼 있고 함수를 순차적으로 호출함으로써 데이터를 조작하는 프로그램이 방식을 절차지향 프로그래밍</font>이라고 합니다.

![[Pasted image 20231115142356.png]]

절차지향 프로그래밍 방식과 달리 <font color="#ffff00">객체지향 프로그래밍은 객체를 정의하는 것에서 시작</font>합니다. 앞서 예시로 사용한 명함 스타트업을 통해 객체지향 프로그래밍에서 말하는 객체라는 개념을 배워봅시다.

먼저 명함을 구성하는 데이터를 생각해보면 명함에는 이름, 이메일, 주소라는 값이 있습니다. 다음으로 명함과 관련된 함수를 생각해보면 기본 명함을 출력하는 함수와 고급 명함을 출력하는 함수가 있을 것 같습니다. <font color="#ffff00">객체지향 프로그래밍이란 아래 그림과 같이 명함을 구성하는 데이터와 명함과 관련된 함수를 묶어서 명함이라는 새로운 객체(타입)을 만들고 이를 사용해 프로그래밍하는 방식을 의미</font>합니다.

![[Pasted image 20231115142606.png]]

[[Python]] 을 통해 명함이라는 객체를 만들어 두면 명함 객체가 정수, 실수, 문자열과 마찬가지로 하나의 타입으로 인식되기 때문에 아래그림과 같이 여러 명의 명함 정보도 쉽게 관리할 수 있습니다. 마치 다섯 개의 정숫값을 변수로 바인딩하는 것처럼 다섯 개의 명함 값을 변수로 바인딩하거나 리스트에 명함이라는 객체를 저장하고 있으면 되는 것입니다.

![[Pasted image 20231115142744.png]]

<font color="#ffff00">클래스를 이용하면 새로운 타입을 만들 수 있다</font>고 했는데, 사실 앞서 배운 정수, 실수, 문자열, 리스트, 튜플과 같은 <font color="#ffc000">기본 자료형과 자료구조도 모두 클래스를 통해 만들어진 타입</font>입니다. 다만 여러분이 만든 것이 아니라 이미 만들어져 있었다는 차이점만 있습니다.

다음코드를 실행해 여러 파이썬 객체에 대한 반환타입을 보면 각 타입 앞에 <font color="#00b050">class</font> 라는 키워드가 있음을 확인할 수 있습니다.

```python
if __name__ == "__main__":
  print(type(3))
  print(type(3.1))
  print(type('3'))
  print(type([]))
  print(type(()))

  def foo():
    pass

  print(type(foo))
```

```
<class 'int'>
<class 'float'>
<class 'str'>
<class 'list'>
<class 'tuple'>
<class 'function'>
```

다른 프로그래밍 언어를 배운 분들에게는 조금 생소하겠지만 파이썬에서는 <font color="#ffff00">함수도 객체</font>입니다. 다음과 같이 `foo` 라는 함수를 정의한 후 해당 함수의 type을 확인해보니 역시 <font color="#00b050">class</font> 키워드가 붙어 있음을 확인할 수 있습니다.

