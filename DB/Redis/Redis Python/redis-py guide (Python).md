
# redis-py guide

[[Redis]] 는 빠르고 효율적인 인메모리 데이터 베이스로, [[Python]] 에서 이를 활용하려면 **redis-py** 라는 클라이언트를 사용해야 합니다. 이 가이드는 redis-py를 설치하고, Redis 데이터베이스에 연결하여 데이터를 저장하고 조회하는 기본적인 방법을 설명합니다.

## Install redis-py

redis-py는 Python에서 Redis와 상호작용하기 위한 공식 클라이언트 라이브러리입니다. 설치 과정은 간단하며, Python의 패키지 매니저인 pip를 사용합니다. 기본 설치 외에, 더 빠른 성능을 위해 **hiredis**를 함께 설치할 수 있습니다. hiredis는 응답 파싱을 최적화하여 성능을 향상시키며, 대부분의 경우 코드 변경 없이 바로 적용 가능합니다.

- 기본 설치 명령어

```
pip install redis
```

- hires와 함께 설치(성능 최적화)

```
pip install redis[hiredis]
```

## Connecting Server Test

Redis 서버에 연결하려면 redis-py의 `Redis` 클래스를 사용합니다. 기본적으로 localhost와 포트 6379를 통해 연결하며, 필요에 따라 추가 연결 옵션을 설정할 수 있습니다. Redis는 응답을 기본적으로 바이트(byte) 형태로 반환하지만 **문자열로 받고 싶다면** `decode_responses=True` 옵션을 설정합니다.

```python
import redis

# localhost:6379에 연결, 응답을 문자열로 디코딩
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
```

### Save and retrieve strings

간단한 키-값 쌍을 저장하고 조회하는 예제입니다. `set` 메서드로 값을 저장하고, `get` 메서드로 값을 조회합니다.

```python
# 키 'foo'에 값 'bar' 저장
r.set('foo', 'bar')  # 반환: True

# 키 'foo'의 값 조회
r.get('foo')  # 반환: 'bar'
```

![[Pasted image 20250624102414.png]]

### Save and retrieve dictionaries

Redis의 해시(hash) 데이터 구조를 사용해 딕셔너리 형태의 데이터를 저장하고 조회할 수 있습니다. `hset` 메서드로 여러 필드를 한 번에 저장하고, `hgetall` 로 전체 데이터를 조회합니다.

```python
# 해시 'user-session:123'에 딕셔너리 데이터 저장
r.hset('user-session:123', mapping={
    'name': 'John',
    'surname': 'Smith',
    'company': 'Redis',
    'age': '29'
})  # 반환: True

# 해시 데이터 전체 조회
r.hgetall('user-session:123')
# 반환: {'name': 'John', 'surname': 'Smith', 'company': 'Redis', 'age': '29'}
```

