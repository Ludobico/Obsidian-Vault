`.size()` 메서드는 <font color="#ffff00">텐서의 크기를 튜플(tuple) 형태로 반환</font>합니다. 예를 들어, 2x3 크기의 2차원 텐서를 생성하고 `.size()` 메서드를 사용하면 (2,3) 이라는 튜플이 반환됩니다. 이 튜플은 각 차원의 크기를 포함하고 있습니다.

```python
import torch

x = torch.tensor([[1,2,3],[4,5,6]])
print(x.size())
```

```
torch.Size([2, 3])
```

`.size()` 메서드는 `shape` 속성과 같은 역할을 합니다. `.size()` 메서드는 메서드로 호출할 수 있지만 `shape` 속성은 인스턴스 변수로서 접근할 수 있습니다.

