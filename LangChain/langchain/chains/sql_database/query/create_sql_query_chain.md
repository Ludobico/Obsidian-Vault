
LLM을 사용해 자연어 질문으로부터 SQL 쿼리를 생성하는 체인을 만들어줍니다.

```
langchain.chains.sql_database.query.create_sql_query_chain
```

> llm -> _[BaseLanguageModel](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.base.BaseLanguageModel.html#langchain_core.language_models.base.BaseLanguageModel "langchain_core.language_models.base.BaseLanguageModel")_,

> db -> _[SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html#langchain_community.utilities.sql_database.SQLDatabase "langchain_community.utilities.sql_database.SQLDatabase")_,

> prompt : [[PromptTemplate]] , optional

> k -> int = 5

## Security note

이 체인은 주어진 데이터베이스에 대해 SQL 쿼리를 자동으로 생성합니다.
`SQLDatabase` 클래스의 `get_table_info` 메서드를 통해 테이블의 컬럼 정보와 샘플 데이터를 가져올 수 있습니다.

**Importance** : 민간함 데이터 유출을 막기 위해 읽기 전용 권한만 부여하고, 접근 가능한 테이블을 필요한 범위로 제한해야 합니다.

자세한 보안 가이드는 [LangChain 보안 문서](https://python.langchain.com/docs/security)를 참고하세요.

