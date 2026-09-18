#retrieval #search

# 검색 파이프라인 — `search_pipeline()`

> 파일: `context/search/pipeline.py:262-348`

한 번의 검색 요청이 실제로 거치는 단계를 순서대로 정리합니다.

```text
1. _build_index_filters()        ACL/테넌트/프로젝트 필터 조립 (03-acl-model.md 참고)
2. strip_stopwords(query)        키워드 검색용 불용어 제거
3. ChunkIndexRequest 조립         랭킹 조절 파라미터 포함
4. search_chunks()               실제 하이브리드(시맨틱+키워드) 검색 + 페더레이션 검색
5. post_query_chunk_censoring()  검색 후 필드 단위 추가 검열 (EE)
6. merge_individual_chunks()     인접 청크를 하나의 "섹션"으로 병합
```

## 1~3단계 — 필터와 랭킹 파라미터 조립

```python
filters = _build_index_filters(..., acl_filters=acl_filters, ...)
query_keywords = strip_stopwords(chunk_search_request.query)

query_request = ChunkIndexRequest(
    query=chunk_search_request.query,
    hybrid_alpha=chunk_search_request.hybrid_alpha,
    recency_bias_multiplier=chunk_search_request.recency_bias_multiplier,
    query_keywords=query_keywords,
    filters=filters,
    limit=chunk_search_request.limit,
)
```

`hybrid_alpha`(시맨틱 vs 키워드 검색 가중치), `recency_bias_multiplier`(최신 문서 가중치)가 요청
객체에 실려서 넘어갑니다 — 즉 **랭킹 전략 자체를 매 요청마다 조절할 수 있게 파라미터화**되어 있습니다.
(실제 랭킹 공식은 `search_chunks`가 위치한 `retrieval/search_runner.py`에 있는데, 이번엔 아직
들어가 보지 않았습니다 — 다음에 더 볼 지점으로 남겨둡니다.)

## 4단계 — `search_chunks()`: 하이브리드 + 페더레이션

```python
retrieved_chunks = search_chunks(
    query_request=query_request,
    user_id=user.id if user else None,
    document_index=document_index,
    db_session=db_session,
    embedding_model=embedding_model,
    prefetched_federated_retrieval_infos=prefetched_federated_retrieval_infos,
)
```

`prefetched_federated_retrieval_infos` 파라미터가 있다는 건, **모든 소스가 사전에 인덱싱되는 게
아니라 일부는 쿼리 시점에 "페더레이션(federated)" 방식으로 실시간 조회된다**는 뜻입니다 — 즉
"미리 색인" 방식과 "쿼리할 때 그 자리에서 조회" 방식이 소스별로 다르게 섞여 있습니다.

## 5단계 — 쿼리 이후의 추가 검열 (또 다른 EE 경계)

```python
censored_chunks = fetch_ee_implementation_or_noop(
    "onyx.external_permissions.post_query_censoring",
    "_post_query_chunk_censoring",
    retrieved_chunks,
)(chunks=retrieved_chunks, user=user)
```

주석이 이유를 설명합니다.

> "For some specific connectors like Salesforce, a user that has access to an object doesn't mean
> that they have access to all of the fields of the object."

[[acl-model]] 에서 본 ACL 필터는 "문서 단위" 권한만 걸러냅니다.** 어떤 소스(예:
Salesforce)는 문서는 볼 수 있어도 특정 필드는 못 보는 경우가 있어서, 검색 결과가 나온 **이후에
필드 단위로 한 번 더 걸러내는 단계**가 따로 있습니다. 이것도 EE 구현체가 없으면 그냥 통과
(no-op)입니다 — 오픈소스 버전에선 이 필드 단위 검열이 적용되지 않습니다.

## 6단계 — `merge_individual_chunks()`: 청크를 "섹션"으로 병합

> 파일: `pipeline.py:145-259`

검색은 문서를 청크(chunk) 단위로 찾아내는데, **같은 문서에서 `chunk_id`가 1씩 차이나는(=바로
옆에 붙어있는) 청크들을 하나의 `InferenceSection`으로 합칩니다.**

```text
문서 X의 청크 5, 6, 7이 검색됐다면
  → 별개의 3개 결과가 아니라, 5~7이 이어붙은 하나의 섹션으로 병합
  → 그 섹션 안에서 원래 가장 먼저(랭킹이 가장 높았던) 청크가 "center_chunk"로 유지되어 순서 결정
```

## 정리

```text
필터(ACL/테넌트) → 하이브리드 검색(+페더레이션) → 필드 단위 검열(EE) → 인접 청크 섹션 병합
```

