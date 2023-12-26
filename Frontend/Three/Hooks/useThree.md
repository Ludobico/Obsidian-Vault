<font color="#00b050">useThree</font> Hook은 React Three Fiber 라이브러리에서 제공되며, Three.js의 상태정보에 접근할 수 있게 해주는 React Hook입니다. 이 Hook을 사용하면 Three.js의 핵심 객체들인 <font color="#00b050">scene, camera, renderer</font> 등에 대한 정보와 캔버스 크기에 대한 측정 값을 얻을 수 있습니다.

여기에 대한 간단한 설명은 다음과 같습니다.

## State properties

> gl -> THREE.WebGLRenderer
- Three.js의 WebGL 렌더러에 대한 참조입니다. WebGLRenderer는 Three.js에서 3D 객체를 렌더링하는데 사용되는 핵심 컴포넌트입니다.

> scene -> THREE.Scene
- Three.js의 씬(scene)에 대한 참조입니다. 씬은 모든 3D 객체, 조명, 카메라 등을 포함하는 컨테이너 역할을 합니다.

> camera -> THREE.PerspectiveCamera
- Three.js의 카메라에 대한 참조입니다. 카메라는 Perspective 카메라로, 화면에 어떻게 그려질지 결정합니다.

> raycaster -> THREE.Raycaster
- Three.js에서 사용되는 기본 레이캐스터(Raycaster)에 대한 참조입니다. 레이캐스터는 <font color="#ffff00">마우스 클릭 등의 이벤트에서 3D 공간상의 객체를 감지하는 데 사용</font>됩니다. 

> pointer -> THREE.Vector2
- 업데이트된, 정규화된, 중앙에 위치한 포인터 좌표를 포함하는 [[THREE.vector2]] 객체입니다. 일반적으로 마우스 이벤트와 함께 사용됩니다.

> mouse -> THREE.Vector2
- 정규화된 이벤트 좌표를 포함하는 [[THREE.vector2]] 객체입니다. `pointer` 를 대신 사용하는 것이 권장됩니다.

> clock -> THREE.Clock
- Three.js에서 사용되는 시스템 클릭에 대한 참조입니다. 시간 기반의 애니메이션 등에서 사용됩니다.

> linear -> bool
- 컬러 스페이스가 선형인 경우 `true` 입니다. 주로 컬러 처리에 관련된 설정입니다.

> flat -> bool
- 톤매핑이 사용되지 않은 경우 `true` 입니다. 주로 조명 및 렌더링 설정과 관련이 있습니다.

> legacy -> bool
- 전역 컬러 관리를 비활성화하는 경우 `true` 입니다. 주로 컬러 매니지먼트와 관련이 있습니다.

> frameloop -> always, demand, never
- 렌더 루프(render loop)의 동작 모드를 설정합니다. 세 가지 모드가 있습니다.
	- 'always' : 항상 렌더링 루프가 동작합니다.
	- 'demand' : 필요할 때만 렌더링 루프가 동작합니다.
	- 'never' : 렌더링 루프를 사용하지 않습니다.

> performance -> { current: number, min: number, max: number, debounce: number, regress: () => void }
- 성능과 관련된 정보를 제공합니다. 싯템 성능에 대한 현재,최소,최대 값과 관련된 속성들이 있습니다.
	- 'current' : 현재 성능 수준을 나타냅니다.
	- 'min' : 최소 성능 수준을 나타냅니다.
	- 'max' : 최대 성능 수준을 나타냅니다.
	- 'debounce' : 성능 업데이트를 얼마나 자주 수행할지를 조절합니다.
	- 'regress' : 성능에 대한 감지 및 업데이트를 수행하는 메서드입니다.

> size -> { width: number, height: number, top: number, left: number, updateStyle?: boolean }
- 캔버스의 크기에 대한 정보를 제공합니다.
- 'width' : 캔버스의 너비를 나타냅니다.
- 'height' : 캔버스의 높이를 나타냅니다.
- 'top', 'left' : 캔버스의 위치를 나타냅니다.
- 'updateStyle' : 스타일을 업데이트할지 여부를 나타내는 bool 값입니다.

