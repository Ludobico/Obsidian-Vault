
[[Redis]] 는 인메모리 데이터베이스로, **빠른 속도와 다양한 데이터 구조를 지원**하여 세션 관리, 캐싱, 실시간 데이터처리 등에 널리 사용됩니다. 특히, [[Python]] 에서 [[Redis Python]] 을 통해 Redis의 기능을 쉽게 활요알 수 있습니다.

아래는 각 메서드에 대한 상세 설명과 예시입니다. 모든 코드는 `decode_responses=True` 를 설정하여 문자열로 응답을 받습니다.

## set

`set` 메서드는 문자열 데이터를 특정 키에 저장합니다. 간단한 키-값 저장소로 사용하거나, 세션 데이터의 단일 값을 저장할 때 사용됩니다.

```python
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

r.set('user:name', 'Alice')
```

![[Pasted image 20250624171754.png]]

## get

`get` 메서드는 지정된 키의 값을 조회합니다. `set` 으로 저장한 데이터를 빠르게 가져올 때 사용됩니다.
키가 없으면 `None` 을 반환합니다.

```python
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

value = r.get('user:name')
print(value)
```

```
Alice
```

## lpush

`lpush` 는 **리스트 데이터 구조에 값을 추가**합니다. 챗 히스토리처럼 <font color="#ffff00">순서가 중요한 데이터를 저장</font>할 때 사용합니다. 

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

message = json.dumps({'question': 'Hi', 'response': 'Hello'})
r.lpush('session:user123:history', message)
```

![[Pasted image 20250624172102.png]]

## lrange

`lrange` 는 리스트에서 지정된 범위의 데이터를 조회합니다. 대화 히스토리 전체를 가져오는 데 사용합니다.
키에 해당하는 리스트의 시작(0) 부터 끝(-1)까지 데이터를 반환합니다.

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

messages = r.lrange('session:user123:history', 0, -1)
for msg in messages:
    print(json.loads(msg))
```

```
{'question': 'Hi', 'response': 'Hello'}
```


## hset

`hset` 은 해시 데이터 구조에서 **키-필드-값 쌍을 저장**합니다. 세션 데이터의 메타데이터(예 : 사용자정보)를 구조화하여 저장할 때 사용합니다.
지정된 키의 해시에 필드-값 쌍을 저장합니다. `mapping` 으로 여러 필드를 한번에 설정 가능합니다.

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

r.hset('user:1000', 'name', 'John Doe')
```

![[Pasted image 20250624172859.png]]

## hgetall

`hgetall` 은 해시의 모든 필드와 값을 조회합니다. 세션 데이터를 한 번에 가져올 때 사용됩니다.
키에 해당하는 해시의 모든 필드-값 쌍을 딕셔너리로 반환합니다.

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

user_data = r.hgetall('user:1000')
print(user_data)
```

```
{'name': 'John Doe'}
```

## del

`del` 은 지정된 키를 삭제합니다. 세션 종료 시 데이터를 정리할 때 사용됩니다.
하나 이상의 키를 삭제하고 삭제된 키를 반환합니다.

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

r.delete('user:1000', 'session:user123:history')
```

## expire

`expire` 는 키에 만료 시간을 설정하여 일정 시간 후 자동 삭제되도록 합니다. 세션 데이터를 일시적으로 유지할 때 사용합니다.
키에 초 단위로 만료 시간을 설정합니다. 이미 설정된 키에만 적용됩니다.

```python
import redis
import json

r.set('temp:session', 'temporary')
r.expire('temp:session', 3600)  # 1시간 후 삭제
```

