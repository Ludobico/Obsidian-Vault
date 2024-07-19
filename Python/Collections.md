---
Created: 2024-07-17
---

## OrderedDict

`OrderedDict` 는 항목이 삽입된 순서를 기억합니다. 이는 `for` 루프와 같은 반복 작업에서 항목이 삽입된 순서대로 반환됨을 의미합니다.

```python
from collections import OrderedDict

d = OrderedDict()
d['a'] = 1
d['b'] = 2
d['c'] = 3

for key, value in d.items():
    print(key, value)
```

```
a 1
b 2
c 3
```

