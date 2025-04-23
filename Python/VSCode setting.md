---
Created: 2025-04-18
---

## Overview

VSCode (혹은 cursorAI) 에서 [[Python]] 을 사용하면서 , **Run python file(shift + n)** 으로 파일을 실행할떄 모듈을 import 하게되는데, 현재 실행되는 파일을 기준으로 실행하게 됩니다.

```python
# module/a.py 에서 실행
from utils import highlight_print
```

그에 따라 <font color="#ffff00">루트프로젝트를 기준</font>으로 `utils` 폴더에있는 `highlight_print` 함수를 실행하면 다음과 같은 에러가 나오게 됩니다.

```bash
ModuleNotFoundError: No module named 'utils'
```

## How to solve it?

이를 해결하려면, **모든 파일의 system path에 루트프로젝트를 입력**하거나 **터미널 실행시 루트프로젝트를 등록** 하는 방법으로 이 문제를 해결할 수 있습니다.

### Regist root project in system path

파이썬에서 실행할 파일에 아래처럼 현재 폴더를 기준으로 루트 프로젝트를 `os` 모듈로 찾아서 system path로 등록합니다.

```python
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
```
