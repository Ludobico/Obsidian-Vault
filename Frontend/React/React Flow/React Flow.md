

![[Pasted image 20250731094222.png]]

React Flow는 [[React]] 기반으로 노드 기반의 UI, 워크플로우, 플로우차트, 다이어그램 등을 쉽게 구축할 수 있는 오픈 소스 라이브러리입니다. 워크플로우 빌더, 노코드 앱, 데이터 시각화, 이미지 처리 등 다양한 용도로 활용되며, 높은 커스터마이징 가능성을 제공합니다. 

## Overview

React Flow는 **노드**와 **엣지**를 사용해 그래프 형태의 인터렉티브 UI를 만드는 데 최적화된 라이브러리입니다. 사용자는 노드를 드래그하고, 엣지를 연결하며, 줌/팬 기능을 통해 동적은 워크플로우를 구현할 수 있습니다.

- **노드 기반 UI**: 노드와 엣지를 통해 그래프 구조를 시각적으로 표현.
- **인터랙티브 기능**: 드래그 앤 드롭, 줌/팬, 노드 선택, 엣지 연결 등 지원.
- **커스터마이징 가능**: 사용자 정의 노드, 엣지, 스타일 등을 쉽게 적용 가능.
- **빠른 렌더링**: 변경된 노드만 다시 렌더링해 성능 최적화.
- **플러그인 컴포넌트**: MiniMap, Controls, Background 등 추가 기능 제공.
- **타입스크립트 지원**: 타입 안정성을 보장하며, Cypress로 테스트됨.
- **활발한 커뮤니티**: MIT 라이선스 하에 무료로 사용 가능하며, Discord와 GitHub를 통해 지원 제공

## Installation

React Flow 를 사용하려면 npm, yarn 또는 pnpm을 통해 설치할 수 있습니다.

```
yarn add @xyflow/react
```

## Example Code

아래는 React Flow를 사용해 간단한 플로우차트를 만드는 예시 코드입니다. 이 코드는 두 개의 노드와 하나의 엣지를 연결하며, MiniMap, Controls, Background 컴포넌트를 포함합니다.

```js
import { useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// 초기 노드와 엣지 설정
const initialNodes = [
  { id: '1', position: { x: 0, y: 0 }, data: { label: 'Node 1' } },
  { id: '2', position: { x: 0, y: 100 }, data: { label: 'Node 2' } },
];
const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

function Flow() {
  // 노드와 엣지 상태 관리
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // 새로운 엣지 연결 시 호출되는 콜백
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
}

export default Flow;
```

- **초기 설정**: initialNodes와 initialEdges로 기본 노드와 엣지를 정의.
- **상태 관리**: useNodesState와 useEdgesState 훅을 사용해 노드와 엣지의 상태를 관리.
- **연결 처리**: onConnect 콜백으로 사용자가 노드를 연결할 때 새로운 엣지를 추가.
- **플러그인 컴포넌트**:
    - <MiniMap />: 그래프의 축소판 뷰 제공.
    - <Controls />: 줌, 팬, 화면 맞춤 등의 컨트롤 버튼 제공.
    - <Background />: 그래프 배경에 격자 또는 점 패턴 추가.
- **스타일**: style.css를 임포트해 기본 스타일 적용.

