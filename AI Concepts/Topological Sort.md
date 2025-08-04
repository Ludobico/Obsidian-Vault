- [[#사용 조건|사용 조건]]
- [[#대표적인 사용 예시|대표적인 사용 예시]]
- [[#Python 예시 코드|Python 예시 코드]]


![[Pasted image 20250804103703.png]]

<font color="#ffff00">위상정렬(Topological Sort)</font> 은 **비순환 방향 그래프에서 노드들을 선형 순서로 정렬하는 알고리즘**입니다.  이 선형 순서는 모든 간선 `u -> v` 에 대해 `u` 가 `v` 보다 앞서도록 정렬됩니다.

즉, **어떤 작업이 다른 작업보다 먼저 수행되어야 하는 순서를 정의**할 때 자주 사용됩니다.

## 사용 조건

- 그래프는 방향성을 가져야 함
- 그래프는 **순환(cycle)이 없어야 함** (DAG : Directed Acyclic Graph)

## 대표적인 사용 예시

- 작업 스케줄링
- 컴파일러의 빌드 순서
- [[ComfyUI]]의 데이터 워크플로우

## Python 예시 코드

```python
from collections import deque, defaultdict

def topological_sort(num_nodes, edges):
    graph = defaultdict(list)
    indegree = [0] * num_nodes

    # 그래프 구성
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    # 진입 차수 0인 노드를 큐에 삽입
    queue = deque([i for i in range(num_nodes) if indegree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != num_nodes:
        raise ValueError("사이클이 존재하여 위상 정렬 불가")
    
    return result
```

```
노드: 0 → 1 → 2
             ↓
            3

위상 정렬 결과: [0, 1, 2, 3] 또는 [0, 1, 3, 2] (여러 결과 가능)
```

