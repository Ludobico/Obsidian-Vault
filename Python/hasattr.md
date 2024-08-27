---
Created: 2024-08-12
---
`hasattr()` 메서드는 [[Python]] 에서 **객체가 특정 속성(attribute)을 가지고 있는지 확인**할 때 사용하는 내장 함수입니다. 이 함수는 두 개의 인자를 받습니다.

1. 확인하고자 하는 객체
2. 확인하고자 하는 속성의 이름(문자열 형태)

`hasattr()` 함수는 해당 객체가 특정 속성을 가지고 있으면 `True`를 반환하고, 그렇지 않으면 `False` 를 반환합니다.

```python
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model

my_car = Car("Toyota", "Corolla")

# 'make' 속성이 있는지 확인
print(hasattr(my_car, 'make'))  # True

# 'year' 속성이 있는지 확인
print(hasattr(my_car, 'year'))  # False
```

