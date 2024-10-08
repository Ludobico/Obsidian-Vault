---
_filters: []
_contexts: []
_links: []
_sort:
  field: rank
  asc: false
  group: false
_template: ""
_templateName: ""
Created: 2024-02-07
---

## What is the Class?
---

지금까지 [[Python]] 에서 클래스라는 개념을 몰라도 프로그래밍하는 데 큰 어려움이 없었습니다. 그러나 API를 사용하려면 반드시 클래스라는 개념을 잘 알아야 합니다. 왜냐하면 API가 대부분 클래스를 이용한 객체지향 프로그래밍 방식으로 개발됐기 때문입니다.

물론, 이처럼 직접적인 이유가 아니더라도 클래스를 이용해 프로그래밍하면 <font color="#ffff00">데이터와 데이터를 조직하는 함수를 하나의 묶음으로 관리</font>할 수 있으므로 복잡한 프로그램도 더욱 쉽게 작성할 수 있습니다. 간단한 예제를 통해 파이썬의 클래스에 대해 본격적으로 배워보겠습니다. 여러분이 명함을 제작하는 스타트업을 창업했다고 생각해 봅시다. 이 스타트업은 고객의 명함을 제작 하기에 앞서 명함에 들어갈 고객 정보(이름, 이메일 주소, 근무지 주소)를 프로그램을 통해 입력받습니다. 파이썬변수를 이용해 다음과 같이 값을 저장할 수 있습니다.

```python
name = "kimyuna"
email = "yunakim@naver.com"
addr = "seoul"
```

다음으로 입력받은 데이터를 사용해 실제로 명함을 출력하는 함수를 하나 만들어 보겠습니다. 다음과 같이 이름, 이메일 주소, 근무지 주소를 함수 인자로 입력받아 포맷에 맞춰 입력값을 출력합니다.

```python
def print_business_card(name : str, email : str, addr : str) -> str:
  print("---------------------------------")
  print("Name : %s" % name)
  print("Email : %s" % email)
  print("Address : %s" % addr)
  print("---------------------------------")


if __name__ == "__main__":
  print_business_card("kimyuna", "yunakim@example.com", "seoul")
```

작성한 함수가 잘 동작하는지 테스트하기 위해 다음과 같이 함수에 인자값을 넣어 함수를 호출해보기 바랍니다. 참고로 함수의 인자로 사용된 변수는 입력된 값을 바인딩하고 있습니다. 결곽값을 보니 작성한 명함 출력 함수가 제대로 동작하는 것을 알 수 있습니다.

```
---------------------------------
Name : kimyuna
Email : yunakim@example.com
Address : seoul
---------------------------------
```

여러분이 창업한 스타트업은 어느덧 입소문을 타서 드디어 회원 수가 한 명에서 두 명으로 늘어났습니다. 회원이 한 명뿐이었던 시절에는 고객 정보를 바인딩할 변수로 `name` `email` `addr` 이라는 이름을 이용했습니다. 이제 회원 수가 한 명 더 늘어서 두 번째 회원에 대한 입력 정보는 다음과 같이 새로운 변수를 사용해 저장했습니다. 참고로 같은 이름의 변수를 사용하면 첫 번째 회원에 대한 정보가 손실됩니다.

```python
name2 = "DusanBack"
email2 = "dusan.back@naver.com"
addr2 = "Kyunggi"
def print_business_card(name : str, email : str, addr : str) -> str:
  print("---------------------------------")
  print("Name : %s" % name)
  print("Email : %s" % email)
  print("Address : %s" % addr)
  print("---------------------------------")


if __name__ == "__main__":
  print_business_card(name2, email2, addr2)
```

두 번째 회원의 고객 정보도 잘 입력받았으므로 명함을 출력해보겠습니다. 앞서 명함을 출력하기 위한 `print_business_card` 라는 이름의 함수를 작성해 뒀기 때문에 두 번째 회원의 명함을 출력하려면 기존에 작성한 명함 출력 함수에 적절한 입력값만 전달하고 함수를 호출하면 됩니다.

```
---------------------------------
Name : DusanBack
Email : dusan.back@naver.com
Address : Kyunggi
---------------------------------
```

아직은 회원이 두 명이라 걱정이 없는데 앞으로 회원 수가 늘어나면 개인 정보를 어떤 방식으로 저장하는 것이 좋을까요? 그리고 고객 정보에 전화번호 및 팩스 번호를 저장하고 이를 출력하는 기능을 추가해야 한다면 어떻게 프로그램을 변경하는 것이 좋을지 생각해 봅시다.

