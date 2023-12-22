react-three-drei 의 [[Orbitcontrols]] 를 사용하여 카메라 시점을 자유롭게 이동시키다가 <font color="#ffff00">특정 순간에 현재 카메라 위치를 고정</font>하고 싶을 때 사용하는 함수에 대해 설명드리겠습니다.

원하는 동작을 수행하기 위해 orbitcontrols의 속성을 받는 <font color="#00b050">ref</font> 와 움직일때마다 <font color="#00b050">log</font>를 출력하는 핸들러 함수가 필요합니다.

### camera handler
```js
const cameraRef = useRef();
  const cameraHandler = () => {
    console.log(cameraRef.current);
  };

<OrbitControls enablePan={true} enableRotate={true} enableZoom={true} ref={cameraRef} onChange={cameraHandler} />
```

마우스를 움직여서 원하는 position을 찾고 log함수의 다음 프로퍼티를 참조합니다.

```js
object :PerspectiveCamera
position : vector3
rotation : vector3
```

참조한 프로퍼티의 position 및 rotation 값을 [[PerspectiveCamera]] 속성에 입력합니다.

