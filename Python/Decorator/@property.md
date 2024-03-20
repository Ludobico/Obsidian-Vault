<font color="#00b050">@peoperty</font> 데코레이터는 파이썬 <font color="#ffff00">클래스의 메서드를 프로퍼티로 변환해주는 기능을 제공</font>합니다. 이를 통해 클래스의 인스턴스에서 해당 프로퍼티을 손쉽게 호출할 수 있으며, 필요한 경우 프로퍼티에 대한 추가적인 로직을 구현할 수 있습니다.

일반적으로 클래스의 속성은 인스턴스 변수로 구현됩니다. 그러나 속성을 직접 호출하면서 추가적인 동작을 수행하거나 값을 수정하는 경우가 있습니다. 이런 경우에 <font color="#00b050">@peoperty</font> 데코레이터를 사용하여 속성을 메서드로 변환할 수 있습니다.

간단한 예를 살펴보겠습니다.

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    @property
    def perimeter(self):
        return 2 * (self.width + self.height)

# 인스턴스 생성
rect = Rectangle(5, 4)

# 속성처럼 호출
print(rect.area)       # 20
print(rect.perimeter)  # 18
```

위의 코드에서 @property 데코레이터를 사용하여 `area()` 와 `perimeter()` 메서드를 프로퍼티로 사용할 수 있게 되었습니다. 그렇기때문에<font color="#ffff00"> ()를 붙이지 않고도 호출할 수 있습니다</font>.

