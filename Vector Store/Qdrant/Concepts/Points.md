
Qdrant의 <font color="#ffff00">Point</font> 는 가장 기본적인 데이터 단위입니다. 각 Point는 다음 세 요소로 구성됩니다.

| 필드      | 설명                                |
| ------- | --------------------------------- |
| id      | 64-bit 정수 또는 UUID                 |
| vector  | 검색 대상 벡터 (dense / sparse / multi) |
| payload | 메타데이터 (JSON 형태)                   |

```json
{
    "id": 129,
    "vector": [0.1, 0.2, 0.3, 0.4],
    "payload": {"color": "red"}
}
```

하나의 [[Vector Store/Qdrant/Concepts/Collections|Collections]] 안에 포함된 포인트들 간에서 vector similarity search 를 수행할 수 있습니다. 모든 포인트는 비동기로 처리되며 두 단계로 진행됩니다.

1. 첫 단계에서는 변경 사항이 Write-Ahead Log(WAL)에 기록됩니다.
2. 이 시점 이후에는, Collection이 종료되어도 데이터가 손실되지 않습니다.

