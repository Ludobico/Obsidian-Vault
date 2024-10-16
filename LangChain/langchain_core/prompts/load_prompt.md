`load_prompt` 기능은 **미리 정의된 프롬프트 템플릿을 로드**하는 데 사용됩니다. 이 기능을 통해 재사용 가능한 프롬프트 템플릿을 쉽게 관리하고 활용할 수 있습니다.

- JSON 또는 YAML 파일에서 프롬프트 템플릿을 로드합니다.

JSON 파일(`prompt.json`)의 내용은 다음과 같습니다.

```JSON
{
    "input_variables": ["name", "task"],
    "template": "안녕하세요, {name}님. 오늘의 {task}를 시작하겠습니다."
}
```

```python
from langchain_core.prompts import load_prompt
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
# 같은 폴더내에 prompt.json 이 있다고 가정
prompt_path = os.path.join(cur_dir, 'prompt.json')
prompt = load_prompt(prompt_path, encoding='utf-8')

variables = {
    "name" : "Alice",
    "task" : "working"
}

formatted_prompt = prompt.format(**variables)
print(formatted_prompt)
```

```
No `_type` key found, defaulting to `prompt`.
안녕하세요, Alice님. 오늘의 working를 시작하겠습니다.
```

