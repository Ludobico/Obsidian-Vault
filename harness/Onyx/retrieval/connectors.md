#connector #acl

# 커넥터 — 문서를 가져오고, 권한을 채우는 방식
모든 커넥터는 `BaseConnector`를 상속하고, 아래 믹스인 중 필요한 걸 조합해서 구현합니다.

```text
LoadConnector              전체 상태를 통째로 가져옴 (최초 색인 등)
PollConnector               시간 범위로 증분 가져옴 (start~end)
SlimConnector                문서 ID만 가져옴 (pruning 등, 권한 정보 불필요)
SlimConnectorWithPermSync   문서 ID + 권한 정보를 함께 가져옴
OAuthConnector               OAuth 인증 흐름 (authorization_url, code_to_token 등)
```

`BaseConnector` 자체에 있는 기본 훅들도 눈에 띕니다.

```python
def validate_perm_sync(self) -> None:
    """이 메서드를 직접 오버라이드하지 말 것 — EE 패키지의 perm_sync_valid.py에 함수를 추가하라"""
    validate_connector_settings_fn = fetch_ee_implementation_or_noop(
        "onyx.connectors.perm_sync_valid", "validate_perm_sync", noop_return_value=None,
    )
    validate_connector_settings_fn(self)
```

권한 동기화 설정이 올바른지 검증하는 것조차 EE 위임 구조로 빠져 있습니다 — [[acl-model]] 에서
본 것과 같은 버전 스위칭 패턴입니다.

## Jira의 실제 구현 — 체크포인트 기반

> 파일: `connectors/jira/connector.py`

**체크포인트 모델** (재시작해도 이어서 받아올 수 있게 하는 상태):

```python
class JiraConnectorCheckpoint(ConnectorCheckpoint):
    all_issue_ids: list[list[str]] = []   # v3(cloud) 엔드포인트용 배치
    ids_done: bool = False
    cursor: str | None = None              # v3 페이지네이션 토큰
    offset: int | None = None              # v2(서버/데이터센터) 페이지네이션 (deprecated)
    seen_hierarchy_node_ids: list[str] = []  # 재시작 시 중복 방지
```

**시간 윈도우 쿼리 — 사소하지만 실제로 겪었을 버그를 피하는 코드:**

```python
def _get_jql_query(self, start, end) -> str:
    # unquoted epoch-ms 사용: 안 그러면 Jira가 naive datetime을
    # API 사용자 프로필의 타임존으로 재해석해버림
    time_jql = f"updated >= {int(start * 1000)} AND updated <= {int(end * 1000)}"
```

주석에 Atlassian 공식 문서 링크까지 남겨서, "왜 이렇게 짰는지"를 근거로 박아뒀습니다.

**권한 포함 여부는 같은 함수를 파라미터로만 분기:**

```python
def load_from_checkpoint(self, start, end, checkpoint) -> CheckpointOutput[JiraConnectorCheckpoint]:
    return self._load_from_checkpoint(jql, checkpoint, include_permissions=False)

def load_from_checkpoint_with_perm_sync(self, start, end, checkpoint) -> CheckpointOutput[...]:
    return self._load_from_checkpoint(jql, checkpoint, include_permissions=True)
```

문서를 가져오는 로직 자체는 하나뿐이고, `include_permissions` 플래그로 "권한까지 조회할지"만
분기합니다 — 권한 조회가 비싸거나(EE API 호출) 불필요한 경로(예: pruning)에서는 끄고 씁니다.

**Slim(ID만) 조회도 같은 패턴:**

```python
def retrieve_all_slim_docs(self, ...):        # pruning용, 권한 조회 없이 ID만
    yield from self._retrieve_all_slim_docs(..., include_permissions=False)

def retrieve_all_slim_docs_perm_sync(self, ...):  # 권한 동기화 패스용
    yield from self._retrieve_all_slim_docs(..., include_permissions=True)
```

페이지네이션은 `JIRA_SLIM_PAGE_SIZE` 단위 배치로 모았다가 다 차면 `yield`하고,
`update_checkpoint_for_next_run()`으로 다음 실행에 이어받을 오프셋/커서를 저장합니다.

## Confluence — 같은 패턴, 대신 "페이지"와 "스페이스" 두 층위

> 파일: `connectors/confluence/access.py`

```python
def get_page_restrictions(confluence_client, page_id, page_restrictions, ancestors, add_prefix=False):
    if not global_version.is_ee_version():
        return None
    ee_fn = fetch_versioned_implementation("onyx.external_permissions.confluence.page_access", "get_page_restrictions")
    return ee_fn(confluence_client, page_id, page_restrictions, ancestors, add_prefix)

def get_page_restrictions_with_per_ancestor_fetch(...):
    """CONFCLOUD-77618 변형: 조상(ancestor) 페이지가 인라인 제한 정보 없이 오는 경우,
    각 조상을 restriction/byOperation으로 개별 조회. 초안(draft)의 403/404는 무시."""

def get_all_space_permissions(confluence_client, is_cloud, add_prefix=False) -> dict[str, ExternalAccess]:
    """스페이스 단위 권한 — 페이지 단위와는 별개 개념"""
```

Jira와 완전히 같은 EE 게이팅 구조(`global_version.is_ee_version()` → `fetch_versioned_implementation`)를
씁니다. 다른 점은 Confluence에는 **"페이지 자체의 제한" + "조상 페이지로부터 상속되는 제한" +
"스페이스 전체 권한"** 이렇게 세 층위가 있고, 조상 페이지가 제한 정보를 인라인으로 안 주는
케이스(CONFCLOUD-77618, 실제 Atlassian 버그 티켓 번호)까지 별도로 처리한다는 점입니다.

## 정리 — 두 커넥터가 공유하는 설계

```text
1. 체크포인트 모델(Pydantic) — 커서/오프셋/완료여부를 명시적 필드로 들고 재시작 안전성 확보
2. "문서 가져오기"와 "권한 포함 여부"를 같은 함수의 bool 파라미터로 분기 (로직 중복 없음)
3. 권한 조회 자체는 항상 EE 경계 뒤 — global_version.is_ee_version() → fetch_versioned_implementation
4. 소스 시스템의 실제 버그/제약(Jira 타임존 재해석, Confluence 조상 페이지 403/404)을
   주석에 근거(공식 문서/버그 티켓 번호)와 함께 남겨서 "왜 이렇게 짰는지"가 코드에 보존됨
```

