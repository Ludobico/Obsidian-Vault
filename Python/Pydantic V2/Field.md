- [[#Parameters|Parameters]]
- [[#Example code|Example code]]
	- [[#Example code#기본값과 필수 코드|기본값과 필수 코드]]
	- [[#Example code#alias 와 JSON 스키마 메타데이터|alias 와 JSON 스키마 메타데이터]]
	- [[#Example code#typing.Annotated 와 Field 결합|typing.Annotated 와 Field 결합]]
- [[#Note|Note]]


[[Pydantic V2]] 의 `Field` 는 `pydantic.Field` 모듈에서 제공되는 함수로, 모델 필드에 대해 다음과 같은 정보를 정의합니다.

- 기본값 : 필드의 기본값 또는 동적 기본값
- 유효성 검사 : 숫자 범위, 문자열 길이, 정규식 등의 제약 조건
- 메타데이터 : JSON 스키마에 반영되는 제목, 설명, 예시 등
- 별칭(alias) : 외부 데이터와 내부 모델의 필드 이름 매핑
- 직렬화/역직렬화 : 필드의 포함/제외 여부 및 동작 제어

## Parameters

Pydantic v2 의 `Field` 는 다양한 파라미터를 지원합니다. 자주 사용되는 매개변수를 다음과 같습니다.

> default

- 필드의 기본값
```python
Field(default="Unknown")
```

> default_factory

- 기본값을 동적으로 생성하는 함수
```python
Field(default_factory=lambda: uuid4().hex)
```

> alias

- 외부 데이터의 필드 이름
```python
Field(..., alias="user_id")
```

> title

- JSON 스키마에 표시될 제목
```python
Field(..., title="User Name")
```

> description

- 필드 설명
```python
Field(..., description="User's full name")
```

> examples

- JSON 스키마에 표시될 예시
```python
Field(..., examples=["John", "Jane"])
```

> gt, ge, lt, le

- 숫자 범위 제약 (초과, 이상, 미만, 이하)
```python
Field(..., gt=0, le=100)
```

> min_length, max_length

- 문자열 길이 제약
```python
Field(..., min_length=3, max_length=50)
```

> pattern

- 정규식 패턴
```python
Field(..., pattern=r'^[a-z]+$')
```

> include, exclude

- 직렬화 시 피드 포함/제외
```python
Field(..., exclude=True)
```

> frozen

- 필드 값 변경 불가
```python
Field(..., frozen=True)
```

> json_schema_extra

- JSON 스키마에 추가 메타 데이터
```python
Field(..., json_schema_extra={"deprecated": True})
```

## Example code

아래는 Pydantic v2 에서 `Field` 를 다양한 상황에서 사용하는 예시 코드들입니다. 각 예시는 특정 기능에 초점을 맞췄습니다.

### 기본값과 필수 코드

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(default="Anonymous", description="사용자의 이름")
    age: int = Field(..., gt=0, le=120, description="사용자의 나이 (필수, 0~120세)")

# 인스턴스 생성
user1 = User(age=25)  # name은 기본값 "Anonymous" 사용
print(user1.model_dump())  # {'name': 'Anonymous', 'age': 25}

# 유효성 검사 실패
try:
    user2 = User(name="John")  # age가 없으므로 에러
except ValueError as e:
    print(e)
```

```
{'name': 'Anonymous', 'age': 25}
1 validation error for User
age
  Field required [type=missing, input_value={'name': 'John'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/missing
```

- `name` 은 기본값이 `"Anonymous"` 로 설정됨
- `age` 는 `...` 로 **필수 필드**이며, 0초과 120 이하로 제한됨
- `model_dump()` 는 v2에서 직렬화를 위해 사용됨

### alias 와 JSON 스키마 메타데이터

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    product_id: int = Field(..., alias="id", title="Product ID", examples=[1001, 1002])
    name: str = Field(..., min_length=1, max_length=100, description="제품 이름")

# 외부 데이터
data = {"id": 1001, "name": "Laptop"}
product = Product(**data)
print(product.model_dump(by_alias=True))  # {'id': 1001, 'name': 'Laptop'}

```

```
{'id': 1001, 'name': 'Laptop'}
```

- `product_id`는 외부 데이터에서 `id`로 들어오지만, 모델 내부에서는 `product_id`로 처리됨(alias).
- `title`과 `examples`는 JSON 스키마에 반영되어 API 문서에 표시됨.
- by_alias=True로 직렬화 시 외부 이름(id)을 사용.

### typing.Annotated 와 Field 결합

```python
from typing import Annotated
from pydantic import BaseModel, Field

class Item(BaseModel):
    price: Annotated[float, Field(gt=0, le=1000, description="가격 (0~1000)")]
    discount: Annotated[float, Field(default=0.0, ge=0, le=100)] = 0.0

item = Item(price=99.99)
print(item.model_dump())  # {'price': 99.99, 'discount': 0.0}
```

```
{'price': 99.99, 'discount': 0.0}
```

- `Annotated`를 사용해 타입(float)과 Field 메타데이터를 결합.
- price는 필수, discount는 기본값 0.0으로 설정.
- V2에서 **Annotated 사용이 권장**됨.

## Note

- **필수 필드와 기본값**: Field(...)는 필수 필드를 나타내며, default나 default_factory가 있으면 필수가 아님.
- **V1과의 차이**: V2에서는 model_dump()를 사용하며, examples와 json_schema_extra가 추가됨.
- **성능**: 복잡한 유효성 검사(예: 정규식)는 성능에 영향을 줄 수 있으므로 적절히 설계.
- **Annotated 활용**: V2에서는 Annotated를 사용해 Field와 타입을 결합하는 것이 권장됨.
- **마이그레이션**: V1에서 V2로 전환 시 Field의 새로운 매개변수와 동작을 확인.

