---
Created: 2024-07-10
---
`importlib` 는 [[Python]] 표준 라이브러리 중 하나로, Python의 **import 매커니즘을 제어하고 사용자 정의 import 동작을 구현할 수 있도록 도와주는 모듈**입니다. 

## Main Functions

### Importing Modules
- `importlib.import_module(name, package=None)` 모듈 이름을 문자열로 받아들여 해당 모듈을 옵옵니다.

```python
import importlib
math_module = importlib.import_module("math")
print(math_module.sqrt(16))
```

```
4.0
```

### Reloading Modules
- `importlib.reload(module)` 이미 가져온 모듈을 다시 로드합니다. 이는 모듈의 소스 코드가 변경된 후 다시 로드하고 싶은 경우 사용합니다.

```python
import importlib
import math

importlib.reload(math)
```

### Creating New Modules
- `importlib.util.module_from_spec(spec)` 모듈 스펙에서 새로운 모듈 객체를 만듭니다.

```python
import importlib
import importlib.util

spec = importlib.util.find_spec("math")
math_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_module)

print(math_module.sqrt(16))
```

```
4.0
```

### Finding Module Specs
- `importlib.util.find_spec(name, package=None)` 모듈의 스펙(spec)을 찾습니다. 이는 모듈 로더, 경로 등의 정보를 포함합니다.

```python
import importlib
import importlib.util

spec = importlib.util.find_spec('math')
print(spec)
```

```
ModuleSpec(name='math', loader=<class '_frozen_importlib.BuiltinImporter'>, origin='built-in')
```

