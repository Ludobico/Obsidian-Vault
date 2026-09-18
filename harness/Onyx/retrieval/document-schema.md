#connectors #data-model

# 커넥터가 실제로 만들어내는 것

> 파일: `connectors/models.py`

지금까지 "커넥터가 문서를 가져온다"고만 말했는데, 실제로 `yield`되는 객체의 정확한 구조입니다

```python
class DocumentBase(BaseModel):
    id: str | None = None
    sections: Sequence[TextSection | ImageSection | TabularSection]
    source: DocumentSource | None = None
    semantic_identifier: str        # UI에 표시될 이름
    metadata: dict[str, str | list[str]]
    doc_updated_at: datetime | None = None
    doc_created_at: datetime | None = None
    primary_owners: list[BasicExpertInfo] | None = None    # 작성자/소유자
    secondary_owners: list[BasicExpertInfo] | None = None  # 담당자 등
    title: str | None = None
    additional_info: Any = None      # 커넥터별 자유 필드
    external_access: ExternalAccess | None = None   # ← ACL이 여기, 문서 단위로 붙음
    parent_hierarchy_raw_node_id: str | None = None  # 이 문서를 담고 있는 폴더/스페이스
    file_id: str | None = None

class Document(DocumentBase):
    id: str            # 필수
    source: DocumentSource  # 필수
```

`Document` 자체는 텍스트를 직접 들고 있지 않고, **여러 개의 `Section`으로 구성**됩니다.

## `Section` — 문서 안의 구조화된 콘텐츠 조각

```python
class Section(BaseModel):
    type: SectionType   # TEXT | IMAGE | TABULAR
    link: str | None
    text: str | None
    image_file_id: str | None
    heading: str | None

class TabularSection(Section):
    """csv/tsv 또는 xlsx 한 시트를 CSV로 변환한 것. 항상 파일 스토어에 스테이징해두고,
    청킹 시점에 한 행씩 스트리밍으로 읽음 — 큰 시트가 메모리에 통째로 안 올라오게."""
    csv_file_id: str
```

**표(테이블) 데이터는 별도 타입(`TabularSection`)으로 분리**되어 있고, 메모리에 다 올리지 않고
파일 스토어에서 스트리밍으로 읽는 구조입니다 — 큰 스프레드시트/표를 다루는 소스(당신 프로젝트의
게임 밸런스 표 같은 것)에 참고할 만한 패턴입니다.

## `SlimDocument` — ID + 권한만 (본문 없음)

```python
class SlimDocument(BaseModel):
    id: str
    external_access: ExternalAccess | None = None
    parent_hierarchy_raw_node_id: str | None = None
    doc_created_at: datetime | None = None
```

[[connectors]] 에서 본 "slim 조회"(pruning, 권한 동기화 전용 패스)가 만들어내는
가벼운 버전입니다 — 본문 없이 존재 확인/권한 확인만 하려는 용도입니다.

## `HierarchyNode` — 폴더/스페이스 같은 "컨테이너" 자체

```python
class HierarchyNode(BaseModel):
    raw_node_id: str
    raw_parent_id: str | None
    display_name: str
    node_type: HierarchyNodeType   # 폴더/스페이스/페이지 등
    external_access: ExternalAccess | None = None   # 컨테이너 자체도 권한을 가짐
```

Confluence의 "스페이스 권한"([[connectors]] 에서 본 `get_all_space_permissions`)이 바로 이
`HierarchyNode.external_access`로 표현됩니다 — 문서 개별 권한과 컨테이너(스페이스/폴더) 권한이
**같은 `ExternalAccess` 타입으로 통일**되어 있다는 점이 핵심입니다.

## 체크포인트의 진짜 베이스는 놀랍도록 단순합니다

```python
class ConnectorCheckpoint(BaseModel):
    has_more: bool   # 이게 전부
```

[[connectors]] 에서 본 `JiraConnectorCheckpoint`(cursor, offset, all_issue_ids...)는
이 한 줄짜리 베이스를 확장한 것입니다 — 커넥터마다 필요한 재시작 상태를 자유롭게 늘려서 씁니다.

## 실패 처리 — 하나가 죽어도 전체가 안 죽는 계약

```python
class DocumentFailure(BaseModel):
    document_id: str
    document_link: str | None = None

class ConnectorFailure(BaseModel):
    failed_document: DocumentFailure | None = None
    failed_entity: EntityFailure | None = None
    failure_message: str
    exception: Exception | None = Field(default=None, exclude=True)  # 로깅용, 직렬화 안 됨

    @model_validator(mode="before")
    def check_failed_fields(cls, values):
        # failed_document와 failed_entity 중 정확히 하나만 있어야 함
```

동기화 도중 개별 문서가 실패해도 `ConnectorFailure` 레코드 하나를 만들어서 계속 진행합니다

## 변경 감지 — 타임스탬프가 부실한 소스를 위한 보험

```python
def content_hash(self) -> str:
    """doc_updated_at을 안 주는 커넥터(예: 웹 커넥터)를 위한 폴백 dedup.
    제목 + 텍스트 섹션 + 이미지 ID + 메타데이터 + 소유자(정렬)를 MD5로 해싱."""
```

소스 시스템이 "최근 수정 시각"을 신뢰할 수 없게 줄 때, **내용 자체를 해싱해서 "진짜 바뀐 문서인지"
판단**하는 폴백입니다.

## 메타데이터 직렬화 — 양쪽이 반드시 같은 포맷을 써야 함

```python
def convert_metadata_dict_to_list_of_strings(metadata: dict) -> list[str]:
    """각 문자열은 'key<SEP>value' 형태. 리스트 값이면 여러 개로 풀림.
    NOTE: 여기서 쓴 포맷팅 방식은 쿼리 필터를 만들 때도 반드시 똑같이 복제해야 한다."""
```

**포팅 시 주의할 지점입니다** — 메타데이터를 저장할 때 쓰는 포맷과, 나중에 그 메타데이터로
필터링(쿼리)할 때 쓰는 포맷이 반드시 일치해야 한다는 게 코드 주석으로 명시돼 있습니다. 저장/조회
양쪽을 다른 코드로 짜면 이 계약이 깨지기 쉬운 지점입니다.

