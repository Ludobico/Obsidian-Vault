---
Created: 2024-07-10
---
`inspect` 는 [[Python]] 의 표준 라이브러리 중 하나로, **프로그램의 객체를 검사하고 분석**하는 데 사용됩니다. 이 라이브러리는 코드 검사, 문서 생성, 객체 탐색 등의 작업을 수행하는데 유용한 기능을 제공합니다.

## Method
---

### inspect.getmembers(object\[, predicate\])

- 이 메서드는 주어진 객체의 모든 멤버를 가져옵니다. 멤버는 메서드, 변수, 내부 클래스 등입니다.

```python
import inspect

class Myclass:
    def __init__(self, x):
        self.x = x
    
    def method1(self):
        pass

    def method2(self, y):
        pass

obj = Myclass(10)

members = inspect.getmembers(obj)

for member_name, member_value in members:
    print(f"Memeber : {member_name}, Type : {type(member_value)}")
```

```
Memeber : __class__, Type : <class 'type'>
Memeber : __delattr__, Type : <class 'method-wrapper'>
Memeber : __dict__, Type : <class 'dict'>
Memeber : __dir__, Type : <class 'builtin_function_or_method'>
Memeber : __doc__, Type : <class 'NoneType'>
Memeber : __eq__, Type : <class 'method-wrapper'>
Memeber : __format__, Type : <class 'builtin_function_or_method'>
Memeber : __ge__, Type : <class 'method-wrapper'>
Memeber : __getattribute__, Type : <class 'method-wrapper'>
Memeber : __getstate__, Type : <class 'builtin_function_or_method'>
Memeber : __gt__, Type : <class 'method-wrapper'>
Memeber : __hash__, Type : <class 'method-wrapper'>
Memeber : __init__, Type : <class 'method'>
Memeber : __init_subclass__, Type : <class 'builtin_function_or_method'>
Memeber : __le__, Type : <class 'method-wrapper'>
Memeber : __lt__, Type : <class 'method-wrapper'>
Memeber : __module__, Type : <class 'str'>
Memeber : __ne__, Type : <class 'method-wrapper'>
Memeber : __new__, Type : <class 'builtin_function_or_method'>
Memeber : __reduce__, Type : <class 'builtin_function_or_method'>
Memeber : __reduce_ex__, Type : <class 'builtin_function_or_method'>
Memeber : __repr__, Type : <class 'method-wrapper'>
Memeber : __setattr__, Type : <class 'method-wrapper'>
Memeber : __sizeof__, Type : <class 'builtin_function_or_method'>
Memeber : __str__, Type : <class 'method-wrapper'>
Memeber : __subclasshook__, Type : <class 'builtin_function_or_method'>
Memeber : __weakref__, Type : <class 'NoneType'>
Memeber : method1, Type : <class 'method'>
Memeber : method2, Type : <class 'method'>
Memeber : x, Type : <class 'int'>
```

### inspect.signature(object)

- 이 메서드는 함수나 메서드의 시그니처(signature)를 가져옵니다. 시그니처는 매개변수의 이름, 디폴트 값, 어노테이션 등을 포함합니다.

```python
import inspect

def my_function(a, b=1, *args, **kwargs):
    pass

signature = inspect.signature(my_function)
print(signature)
```

```
(a, b=1, *args, **kwargs)
```

