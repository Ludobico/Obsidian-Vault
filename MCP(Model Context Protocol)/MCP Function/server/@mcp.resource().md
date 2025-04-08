
<font color="#ffff00">resource</font> 는 [[FastMCP]] 서버에서 LLM이나 클라이언트가 **데이터를 가져올 수 있도록 정의된 엔트리 포인트**입니다. `@mcp.resource()` 데코레이터를 사용해 함수를 "리소스"로 등록하며, 다음과 같은 목적을 가집니다.

1. **데이터 제공** : 정적 데이터(설정)나 동적 데이터(사용자 프로필)를 반환
2. **읽기 전용** : REST API의 GET 메서드처럼, 데이터를 조회하는데 초점이 맞춰져 있으며 상태 변경이나 복잡한 계산을 피함.
3. **LLM 통합** : LLM이 "앱 설정이 뭐야?" 또는 "사용자 프로필을 보여줘" 같은 요청을 처리할 때 호출됨.

## Parameters

- 정적 경로
	- 예 : `@mcp.resource("config://app")`
	- 함수는 인자 없이 호출

- 동적 경로
	- 예 : `@mcp.resource("users://{user_id}/profile"`
	- 함수는 `user_id` 를 인자로 받음



## Example

```python
@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"
```

- `@mcp.resource("config://app")` 리소스의 식별자로 "config://app" 을 사용, 이는 **URL 형식으로, 리소스의 경 로 또는 이름**을 나타냄

```python
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"
```

