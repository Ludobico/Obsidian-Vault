
[[Python]] 의 `@dataclass` 는 **데이터를 보관하는 클래스** 를 쉽게 만들 수 있게 해주는 기능입니다. 이 데코레이터를 사용하면 자동으로 <font color="#ffff00">init(), repr(), eq()</font> 등의 특수 메서드를 생성해줍니다.

```python
from dataclasses import dataclass

@dataclass
class Student:
    name : str
    age : int
    grade : float = 5

student = Student("홍길동", 20, 4.2)

print(student)
print(student.name)
```

```
Student(name='홍길동', age=20, grade=4.2)
홍길동
```

