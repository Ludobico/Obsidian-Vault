
[Smithery](https://smithery.ai/server/@smithery-ai/server-sequential-thinking)

[github](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)

Sequential Thinking MCP Server는 구조적인 사고 방식을 통해 **복잡한 문제를 동적으로 해결하는 AI 기반 서버**입니다.

- 문제를 작은 단계로 나누어 분석
- 이해가 깊어질수록 생각을 수정하고 개선
- 다른 해결 경로를 탐색하며 유연한 문제 해결 가능

## Tool : sequential_thinking

순차적 사고 과정을 상게하게 조정하여 분석 및 문제 해결을 도와주는 도구

### Inputs

| 파라미터              | 타입       | 설명                |
| ----------------- | -------- | ----------------- |
| thought           | string   | 현재 사고 단계          |
| nextThoughtNeeded | boolean  | 다음 사고 단계가 필요한지 여부 |
| thoughtNumber     | interger | 현재 사고 단계 번호       |
| totalThoughts     | interger | 전체 예상 사고 단계 수     |
| isRevision        | boolean  | 이전 사고 과정을 수정하는 여부 |
| revisesThought    | interger | 수정할 사고 단계 번호      |
| branchFromThought | interger | 분기할 사고 단계 번호      |
| branchId          | string   | 분기 식별자            |
| needsMoreThouhts  | boolean  | 추가 사고 단계가 필요한지 여부 |


