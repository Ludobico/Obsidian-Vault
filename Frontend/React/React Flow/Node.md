- [[#Overview|Overview]]
- [[#Main features|Main features]]
- [[#Node types|Node types]]

## Overview

노드는 [[React Flow]] 에서 **그래프의 개별 요소**를 나타내며, 위치, 데이터, 스타일 등을 포함하는 객체입니다. React Flow 에는 기본적으로 `default` `input` `output` `group` 등의 내장 노드 타입을 제공하며, 사용자가 커스텀 노드를 정의해 원하는 기능과 UI를 구현할 수 있습니다. 노드는 다음과 같은 속성을 가집니다.

> id
- 노드를 고유하게 식별하는 문자열

> position
- 노드의 x, y 좌표

> data
- 노드에 저장할 사용자 정의 데이터 (라벨, 값 등)

> type
- 노드의 타입 (기본 또는 커스텀)

> style
- CSS 스타일 객체로 노드의 외관 커스터마지이

> selected, hidden, draggable
- 선택 여부, 표시 여부, 드래그 가능 여부 등

## Main features

React Flow 의 노드는 다양한 기능을 제공하며, 아래는 주요 기능과 관련된 설명입니다.

1. 드래그 앤 드롭
노드는 기본적으로 드래그 가능하며, 사용자가 캔버스 내에서 자유롭게 이동할 수 있습ㄴ디ㅏ. `draggable` 속성을 `false` 로 설정하면 드래그를 비활성화 할 수 있습니다.

2. 선택 및 다중선택
노드는 클릭하거나 키보드(Enter/space)로 선택할 수 있으며, Shift 키를 사용해 여러 노드는 선택할 수 있습니다. 선택된 노드는 `selected` 속성이 `true` 로 설정되며, 이를 통해 스타일링이나 로직을 커스터마이징할 수 있습니다.

3. 삭제
선택된 노드는 Delete/Backspace 키로 삭제 가능합니다. `onDelete` 또는 `onBeforeDelete` 훅을 사용해 삭제동작을 커스터마이징 할 수 있습니다.

4. 커스텀 노드
노드는 [[React]] 컴포넌트로 정의되며, `nodeTypes` prop을 통해 커스텀 노드를 등록할 수 있습니다. 이를 통해 입력 필드, 버튼, 차트 등 원하는 UI와 기능을 추가할 수 있습니다.

5. 핸들
핸들은 노드의 연결 지점으로, 엣지를 연결하는 데 사용됩니다. `sourcePosition` 과 `targetPosition` 으로 핸들의 위치를 지정하며, 커스텀 노드에서 `<Handle>` 컴포넌트를 사용해 커스터마이징 가능합니다.

6. 서브 플로우
노드는 다른 노드의 자식 노드로 설정할 수 있으며, `parentId` 를 지정해 그룹화된 서브 플로우를 구성할 수 있습니다. 부모 노드 이동 시 자식 노드도 함께 이동합니다. `extent : 'parent'` 로 자식 노드가 부모 밖으로 벗어나지 않도록 제한할 수 있습ㄴ디ㅏ.

7. 노드 툴바
`<NodeToolbar>` 컴포넌트를 사용해 노드에 툴바를 추가할 수 있습니다. 툴바는 노드 선택 시 나타나며, 버튼이나 기타 UI 요소를 포함합니다.


## Node types

[[React Flow]] 는 기본적으로 4가지 내장 노드 타입을 제공합니다.

1. default
	1. 기본 노드 타입으로, 일반적인 직사각형 노드를 렌더링합니다.
	2. 데이터의 `label` 속성을 표시하며, 양방향 핸들을 기본 지원합니다.
	3. 커스텀 스타일이나 동작이 필요 없는 기본적인 경우에 적합합니다.

2. input
	1. 입력 전용 노드로, 타겟 핸들만 제공합니다. (소스 핸들 없음)
	2. 워크플로우의 시작 지점이나 입력 노드로 사용합니다.

3. output
	1. 출력 전용 노드로, 소스 핸들만 제공합니다. (타겟 핸들 없음)
	2. 워크플로우의 끝 지점이나 출력 노드로 사용합니다.

4. group
	1. 다른 노드를 포함할 수 있는 그룹 노드로, 서브 플로우를 구성할 때 사용합니다.
	2. `style` 속성으로 크기를 지정하며, `parentId` 를 통해 자식 노드를 그룹 내에 포함합니다.
	3. 자식 노든는 `extent : 'parent'` 로 부모 노드 내부에 제한될 수 있습니다.

```js
const initialNodes = [
  { id: '1', type: 'input', data: { label: '왼쪽 (Input)' }, position: { x: 50, y: 50 } },
  { id: '2', type: 'default', data: { label: '오른쪽 (Default)' }, position: { x: 250, y: 50 } },
  { id: '3', type: 'output', data: { label: '출력 (Output)' }, position: { x: 150, y: 150 } },
  { id: '4', type: 'group', position: { x: 0, y: 250 }, style: { width: 200, height: 150 } },
  { id: '5', type: 'custom', parentId: '4', data: { label: '그룹 내 노드 (Custom)' }, position: { x: 50, y: 50 }, extent: 'parent' },
];

export default initialNodes;
```

```js
const initialEdges = [
  { id: 'e1-2', source: '1', target: '2' },
  { id: 'e2-3', source: '2', target: '3' },
];

export default initialEdges;
```

```js
import { Handle, Position } from '@xyflow/react';

function CustomNode({ data }) {
  const onChange = (evt) => {
    console.log('입력값:', evt.target.value);
  };

  return (
    <div style={{ padding: 10, background: '#fff', border: '1px solid #000', borderRadius: 3 }}>
      <Handle type="target" position={Position.Top} />
      <div>
        <label>{data.label}</label>
        <input
          className="nodrag"
          onChange={onChange}
          placeholder="텍스트 입력"
          style={{ display: 'block', marginTop: 5 }}
        />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default CustomNode;
```

```js
import { useCallback } from 'react';
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  MiniMap,
  Controls,
  Background,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import initialNodes from './nodes';
import initialEdges from './edges';
import CustomNode from './CustomNode';

// 노드 타입 정의
const nodeTypes = {
  custom: CustomNode,
};

const FlowCanvas = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
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
        nodeTypes={nodeTypes}
        colorMode="dark"
        fitView
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
};

export default FlowCanvas;
```

![[Pasted image 20250731103057.png]]
