<font color="#00b050">PerspectiveCamera</font> 는 인간의 눈이 깊이를 인식하는 방식을 투영한 <font color="#ffff00">원근감을 반영한 카메라 유형</font>입니다.

React Three Fiber 컴포넌트에서 Perspective Camera를 사용하는 간단한 예시입니다.

```js
import React from 'react';
import { Canvas } from 'react-three-fiber';
import { PerspectiveCamera } from 'drei';

const MyScene = () => {
  return (
    <Canvas>
      <PerspectiveCamera
        position={[0, 0, 5]}  // 3D 공간에서 카메라 위치 설정
        fov={75}              // 시야각 설정
        near={0.1}            // 근 클리핑 평면 설정
        far={100}             // 원격 클리핑 평면 설정
      />
      {/* 다른 3D 장면 요소들은 이곳에 추가 */}
    </Canvas>
  );
};

export default MyScene;
```

## <font color="#ffc000">Properties</font>

카메라 클래스에 속하는 프로퍼티들에 대한 내용입니다. 이러한 프로퍼티들은 3D 장면엣 카메라의 시각적인 특성을 설정하고 제어하는 데 사용됩니다.

> aspect -> float

가로 세로 비율, 일반적으로 캔버스의 너비/ 캔버스의 높이입니다. 기본값은 1입니다.

> far -> float

카메라의 far 속성은 시야(fov) 범위 내에서 물체를 렌더링하는 최대 거리를 결정합니다. 렌더링할 수 있는 거리가 far 값보다 큰 물체는 렌더링되지 않습니다.

> near -> float

카메라로 볼 수 있는 최소 거리를 나타내며 기본값은 0.1 입니다.

> fov -> float

시야각(Field of View)를 나타냅니다. 시야각은 카메라가 얼마나 넓은 영역을 볼 수 있는지를 결정하는 각도입니다.

이 각도는 수직 또는 수평 방향으로 설정될 수 있으며, 일반적으로는 수직 시야각이 사용됩니다.
기본값은 50 입니다.

> makeDefault -> bool

카메라를 시스템 기본값으로 등록할지 여부를 나타냅니다. `true` 로 설정하면, React Fiber는 이 카메라로 렌더링을 시작합니다.

> manual -> bool

이 속성을 `true`로 설정하면 카메라가 반응성을 중지하고 가로 세로 비율을 직접 계산해야 합니다.

> frames -> int

렌더링할 프레임 수를 나타냅니다. 기본값은 0이며, 0으로 설정하면 렌더링이 한 번만 수행됩니다.

> resolution -> int

프레임 버퍼 객체(FBO)의 해상도를 나타냅니다. 기본값은 256입니다.

> envMap -> THREE.Texture

기능적인 용도로 사용될 수 있는 환경 맵을 지정합니다.

## <font color="#ffc000">Methods</font>

