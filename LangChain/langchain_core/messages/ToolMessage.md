`ToolMessage`  는 [[LangChain/LangChain|LangChain]] 에서 Tool 실행 결과를 모델에 전달하기 위한 특별한 메시지 타입입니다. 

```python
from langchain_core.messages import ToolMessage

# 기본 사용법
simple_tool_message = ToolMessage(
    content='42',  # 도구 실행 결과
    tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL'  # 도구 호출 식별자
)
```

2. 주요 구성 요소:

- content: 도구 실행의 주요 결과값
- tool_call_id: 도구 호출을 식별하는 고유 ID
- artifact: 전체 도구 출력을 저장하는 선택적 필드

3. 고급 사용 예시:

```python
# 복잡한 도구 출력 처리
tool_output = {
    "stdout": "그래프에서 x와 y 사이의 상관관계를 확인할 수 있습니다...",
    "stderr": None,
    "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."}
}

detailed_tool_message = ToolMessage(
    content=tool_output["stdout"],  # 주요 출력 내용
    artifact=tool_output,           # 전체 출력 데이터
    tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL'
)
```

4. 주요 용도:

- 도구 실행 결과 전달: 도구가 생성한 결과를 LLM에 전달
- 병렬 도구 호출 관리: tool_call_id를 통해 여러 도구 호출을 추적
- 복잡한 출력 처리: 텍스트, 이미지 등 다양한 형태의 출력 처리

5. 특징:

- BaseMessage 상속: 기본 메시지 기능 모두 사용 가능
- 구조화된 데이터 전달: content와 artifact를 통한 유연한 데이터 전달
- 추적 가능성: tool_call_id를 통한 도구 호출 추적

6. 활용 시나리오:

```python
# 데이터 분석 도구 결과 전달
analysis_result = ToolMessage(
    content="데이터 분석 결과: 평균값은 75.3입니다",
    tool_call_id="analysis_001",
    artifact={"raw_data": [72, 75, 79], "statistics": {"mean": 75.3}}
)

# 이미지 생성 도구 결과 전달
image_result = ToolMessage(
    content="이미지가 성공적으로 생성되었습니다",
    tool_call_id="image_gen_001",
    artifact={"image_data": "base64...", "format": "png"}
)
```


ToolMessage는 LangChain에서 도구와 LLM 사이의 효과적인 통신을 가능하게 하며, 특히 복잡한 도구 체인을 구축할 때 중요한 역할을 합니다.