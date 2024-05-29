---
Created: 2024-03-15
---

<font color="#00b050">**kwargs</font> 는 [[Python]] 함수의 인자로 사용되는 특별한 형태의 매개변수입니다. kwargs는 **keyword arguments** 의 약자입니다.

일반적으로 함수를 정의할 때, <font color="#ffff00">인자의 개수나 종류를 미리 정해놓지 않고 유연하게 다양한 인자를 전달하고 싶을 때 사용</font>합니다. 보통 이러한 상황은 함수가 다양한 설정 옵션을 가지거나, 다른 함수에 전달할 인자를 동적으로 선택할 때 발생합니다.

<font color="#00b050">**kwargs</font> 는 딕셔너리 형태로 전달되며, 함수 내에서 사용되는 인자의 이름과 그에 해당하는 값을 담고 있습니다. 이를 통해 함수 내에서 해당 인자들을 활용할 수 있습니다.

> ** kwargs 를 사용할때 반드시 가장 뒤쪽에 와야 합니다. 
> ex) def custom_print(a, b, * args, ** kwargs):


아래는 간단한 예시입니다.

```python
def print_info(**kwrags):
  for key, value in kwrags.items():
    print(f"{key} : {value}")

print_info(name="Alice", arg=30, city="New York")
```

```
name : Alice
arg : 30
city : New York
```

