여기에서는 [[LangGraph]] 로 정의된 워크플로우를 컴파일하고, **Mermaid 다이어그램으로 시각화**합니다.

```python
from PIL import Image
import io
import os

# 워크플로우 컴파일
workflow = Workflow()  # 실제 워크플로우 정의에 따라 수정
graph = workflow.compile()
graph_data = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(graph_data))

image.show()
```

