
[[Qdrant]] 의 <font color="#ffff00">Collection</font> 은 **검색 단위를 정의하는 논리적 컨테이너**입니다. 각 Collection은 동일한 차원과 metric을 갖는 벡터 집합을 저장하며, 벡터화 함께 payload(metadata)를 포함할 수 있습니다.

하나의 <font color="#ffff00">point</font> 는 여러 벡터를 가질 수도 있는데, 이 경우 각 벡터에 이름을 붙여 서로 다른 차원이나 metric을 적용할 수 있습니다.
예를 들어, `image` 벡터는 Dot-product 기반, `text` 벡터는 Cosine 기반으로 설정 가능합니다.

Qdrant는 다음 함수를 지원합니다.

- [[dot product]]
- [[Cosine similarity]]
- [[L2, Euclidean Distance]]
- [[L1, Manhattan Distance]]

이때, Cosine은 업로드 시 자동으로 정규화(normalization)되어 저장됩니다.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
)
```

## Collection components

Collection은 단순히 벡터를 저장하는 공간이 아니라, **검색 효율과 저장 최적화 전략을 제어**하는 설정을 포함합니다. 다음과 같은 옵션을 통해 튜닝할 수 있습니다.

- `hnsw_config` : 인덱싱 전략
- `wal_config` : Write-Ahead Log 설정
- `optimizers_config` : 메모리 및 디스크 최적화 설정
- `on_disk_payload` : 대형 payload 의 디스크 저장 여부
- `quantization_config` : 벡터 양자화 세부 설정
- `shard_number` : 분산 환경에서의 샤드 개수
- `strict_mode_config` : 데이터 검증 모드

기본적으로 모든 벡터는 RAM에 적재되며, `on_disk = true` 설정 시 memmap 기반 I/O로 대용량 벡터를 다룰 수 있습니다.


## Vector Datatypes

Qdrant V1.9.0 부터 다양한 벡터 데이터 타입을 지원합니다.
특히 일부 임베딩 벤더는 <font color="#ffff00">사전 양자화(pre-quantized)</font> 된 벡터를 제공합니다. 이러한 경우, Qdrant의 `uint8` 타입을 사용하면 불필요한 변환 과정 없이 저장하고 검색할 수 있습니다.

`uint8` 벡터는 메모리 사용량을 줄이고 검색 속도를 높이지만, 8비트 정밀도로 인해 미세한 유사도 차이는 손실될 수 있습니다.

양자화 과정에 대해서는 [[Quantization]]에서 확인할 수 있습니다.

각 요소는 `0~255` 범위의 <font color="#ffff00">unsigned 8-bit integer</font> 로 표현됩니다.

```python
client.create_collection(
    collection_name="example",
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
        datatype=models.Datatype.UINT8,
    ),
)

```

### Sparse Vectors

Qdrant v1.7.0 부터 희소벡터(sparse vectors)를 지원합니다.
희소벡터는 항상 named vector로 선언되어야 하며, dense vector와 이름이 중복되면 안됩니다.

예를 들어, `text` 라는 희소 벡터를 추가하려면 아래의 코드로 구현됩니다.

```python
client.create_collection(
    collection_name="docs",
    vectors_config={},
    sparse_vectors_config={
        "text": models.SparseVectorParams(),
    },
)

```

