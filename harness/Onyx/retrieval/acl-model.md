#acl #security 

# ACL 모델 — 문서와 사용자가 "토큰 집합"으로 매칭되는 구조

## 핵심 아이디어

[[Onyx]]의 ACL은 권한 체크 로직이 아니라 **집합 교집합 문제**로 설계되어 있습니다. `_get_acl_for_user`의 docstring이 이걸 정확히 요약합니다.

> "The user should have access to a document if **at least one entry in the document's ACL
> matches one entry in the returned set**."

즉 "문서가 가진 ACL 토큰 집합"과 "사용자가 가진 ACL 토큰 집합" 사이에 **겹치는 토큰이 하나라도
있으면 접근 허용**입니다. 권한 로직이 if/else 분기가 아니라 벡터/키워드 인덱스의 필터 조건(배열
overlap)으로 그대로 내려갑니다.

## 토큰 네임스페이스 (충돌 방지용 prefix)

> 파일: `access/utils.py`

```python
def prefix_user_email(user_email):      return f"user_email:{user_email}"
def prefix_user_group(user_group_name): return f"group:{user_group_name}"
def prefix_external_group(ext_group_name): return f"external_group:{ext_group_name}"
def build_domain_group_id(domain):      return f"domain:{domain.lower()}"
```

```text
PUBLIC_DOC_PAT = "PUBLIC"   (configs/constants.py)
```

각 prefix는 "이메일 문자열과 그룹 이름이 우연히 같아서 잘못 매칭되는 사고"를 막기 위한 것입니다.
`external_group:`은 소스 시스템(Jira, Confluence 등)에서 온 그룹임을 별도로 표시하고,
`build_ext_group_name_for_onyx`는 소스별로 그룹 이름 앞에 `{source}_`를 더 붙여서 "Jira의
`engineering` 그룹"과 "Confluence의 `engineering` 그룹"이 서로 충돌하지 않게 합니다.

## 문서 쪽 — `DocumentAccess.to_acl()`

> 파일: `access/models.py:160-199`

```python
class DocumentAccess(ExternalAccess):
    user_emails: set[str | None]        # Onyx 자체 사용자
    user_groups: set[str]                # Onyx 자체 그룹
    external_user_emails: set[str]       # 커넥터(Jira/Confluence)가 채운 값
    external_user_group_ids: set[str]    # 커넥터가 채운 값
    is_public: bool

    def to_acl(self) -> set[str]:
        acl_set = set()
        for user_email in self.user_emails: acl_set.add(prefix_user_email(user_email))
        for group_name in self.user_groups: acl_set.add(prefix_user_group(group_name))
        for external_user_email in self.external_user_emails: acl_set.add(prefix_user_email(external_user_email))
        for external_group_id in self.external_user_group_ids: acl_set.add(prefix_external_group(external_group_id))
        if self.is_public: acl_set.add(PUBLIC_DOC_PAT)
        return acl_set
```

`external_user_emails`/`external_user_group_ids`는 이전에 본 `ExternalAccess`(커넥터가 만드는
값)에서 그대로 내려온 필드입니다. **Onyx 자체 사용자/그룹과 외부 시스템(Jira 등) 사용자/그룹이
결국 같은 형식의 문자열 토큰 하나의 집합으로 합쳐집니다** — 그래서 검색 엔진 입장에서는 "이 문서가
어느 소스에서 왔는지" 신경 쓸 필요 없이 그냥 문자열 배열 매칭만 하면 됩니다.

## 사용자 쪽 — `get_acl_for_user`

> 파일: `access/access.py:114-142`

```python
def _get_acl_for_user(user: User, db_session: Session) -> set[str]:
    if user.is_anonymous:
        return {PUBLIC_DOC_PAT}
    return {
        prefix_user_email(user.email),
        *(prefix_user_email(email) for email in user.prior_emails),
        PUBLIC_DOC_PAT,
    }

def get_acl_for_user(user, db_session=None) -> set[str]:
    versioned_acl_for_user_fn = fetch_versioned_implementation("onyx.access.access", "_get_acl_for_user")
    return versioned_acl_for_user_fn(user, db_session)
```

- 이전 이메일(`prior_emails`)까지 포함하는 이유: "주소 A로 색인된 문서가, 소스 시스템이 아직
  재동기화 안 됐을 때도 계속 A로 남아있는 경우"를 대비한 것 — 이메일이 바뀐 뒤에도 옛 주소로
  달린 ACL을 놓치지 않게 하려는 설계입니다 (단, 그 주소가 다른 사람에게 넘어가면 즉시 무효화됩니다).
- `get_acl_for_user`는 **버전 스위칭(`fetch_versioned_implementation`) 구조**입니다 — 이게 EE
  경계입니다. 아래에서 다룹니다.

## 쿼리 시점 적용

```text
build_access_filters_for_user(user, db_session)   # preprocessing/access_filters.py
  → get_acl_for_user(user, db_session)
  → IndexFilters.access_control_list = [...]        # 검색 파이프라인의 필터 필드
  → document_index.keyword_retrieval(filters=...)   # 실제 인덱스 조회 시 필터로 적용
```

문서의 `to_acl()` 결과와 사용자의 `get_acl_for_user()` 결과가 **같은 prefix 함수로 만들어졌기
때문에** 문자열 비교만으로 매칭이 됩니다 — 이게 이 설계의 핵심이자, prefix를 통일해야만 하는
이유입니다.

## 오픈소스 버전의 한계 — 두 개의 EE 경계

**① 사용자 쪽 — 그룹 확장이 없습니다.** `_get_acl_for_user`의 오픈소스 기본값은 `이메일 + PUBLIC`
뿐입니다. "이 사용자가 속한 그룹" 자체를 계산하는 로직이 여기 없습니다 — SSO/디렉토리에서 그룹
멤버십을 가져와 `prefix_user_group`/`prefix_external_group` 토큰을 추가하는 건 버전 스위칭으로
교체되는 구현체(EE)의 몫으로 보입니다.

**② 문서 쪽 — 인덱싱 중 권한 자체를 안 가져옵니다.**

```python
def source_should_fetch_permissions_during_indexing(source: DocumentSource) -> bool:
    return cast(bool, fetch_ee_implementation_or_noop(
        "onyx.external_permissions.sync_params",
        "source_should_fetch_permissions_during_indexing",
        False,   # ← EE 구현체가 없으면 기본값 False
    ))
```

EE 구현체가 없으면 **어떤 소스에 대해서도 기본값이 `False`** — 즉 인덱싱 중에 문서별 권한을 가져오는
시도 자체를 안 합니다. (Jira의 `get_project_permissions`가 EE 전용이라는 것도 이 경계의 한 사례입니다.)

## 정리

```text
매칭 메커니즘(교집합 방식, prefix 스킴, IndexFilters로 내려가는 구조) = 오픈소스, 완전히 동작함
사용자 그룹 확장                                                    = EE 전용
소스별 문서 권한 수집(대부분의 커넥터)                                = EE 전용 (기본 False)
```

