> vec2
- `vec2` 는 2D 벡터를 나타내는 GLSL 데이터 타입입니다. 이는 두 개의 요소를 갖는 벡터를 정의합니다.

> vec3
- `vec3` 는 3D 벡터를 나타내는 데이터 타입입니다. 이 벡터는 일반적으로 (x,y,z) 와 같은 3개의 구성 요소를 가집니다. 이것은 주로 공간 좌표나 컬러 값을 표현하는 데 사용됩니다. 예를 들면 `vec3(1.0, 0.0, 0.0)` 은 빨간색을 나타내는 3D 벡터입니다.

> vec4
- `vec4` 는 4D 벡터값을 나타내는 데이터 타입입니다. 4개의 구성 요소를 가지며 일반적으로 (x,y,z,w) 와 같이 사용됩니다. 여기서 w는 보통 투명도 값을 나타냅니다. 예를 들면 `vec4(1.0, 0.0, 0.0, 1.0)`은 빨간색이며 완전히 불투명한 4D 벡터입니다.

> uniform
- `uinform` 예약어는 이 변수가 어떤 유형의 외부에서 제공되는 전역 변수임을 나타냅니다. `uniform` 변수는 fragment 쉐이더나 vertex 쉐이더에서 동일한 값이어야 합니다. 사용자가 쉐이더 외부에서 이러한 값을 설정할 수 있습니다.

> gl_FragColor
- `gl_FragColor`은 GLSL(OpenGL Shading Language) 프래그먼트 쉐이더에서 사용되는 특별한 내장 변수로, <font color="#ffff00">프래그먼트의 최종 색상</font>을 나타냅니다. 프래그먼트 쉐이더는 각 픽셀(프래그먼트)에 대한 색상 값을 계산하고 이를 `gl_FragColor`에 할당하여 렌더링합니다.
- 일반적으로 gl_FragColor은 vec4 형식의 값을 가지며, 이는 일반적으로 RGBA(빨강, 초록, 파랑, 투명도)를 나타냅니다. 예를 들어, 다음은 빨간색을 나타내는 vec4 값을 gl_FragColor에 할당하는 코드입니다
```glsl
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); // 빨간색 (RGBA)
}
```

> gl_FragCoord
- `gl_FragCoord` 는 GLSL fragment 쉐이더에서 사용되는 내장 변수로, <font color="#ffff00">현재 픽셀의 화면 좌표</font>를 나타냅니다. 이 변수는 `vec4` 형식으로 x,y,z 그리고 $\frac{1}{w}$ 값을 포함합니다.
- `gl_FragCoord.x`: 현재 프래그먼트의 화면 좌표 X 값
- `gl_FragCoord.y`: 현재 프래그먼트의 화면 좌표 Y 값
- `gl_FragCoord.z`: 현재 프래그먼트의 화면 좌표 Z 값 (깊이 값, 일반적으로 0.0에서 1.0 사이의 값)
- `gl_FragCoord.w`: 현재 프래그먼트의 화면 좌표에 대한 역수 (1.0 / W)
프래그 쉐이더에서 `gl_FragCoord`를 사용하면 각 픽셀의 화면 좌표에 따라 다른 동작을 수행할 수 있습니다. 예를 들어, 이를 사용하여 픽셀의 좌표에 따라 색상을 변화시키거나 특정 패턴을 생성하는 등의 작업을 수행할 수 있습니다.

아래는 간단한 예제 코드입니다. 여기서는 `gl_FragCoord.x` 값을 사용하여 빨간색을 좌에서 우로 선형 그라데이션하는 색상을 만듭니다:
```glsl
void main() {
    float normalizedX = gl_FragCoord.x / u_resolution.x; // u_resolution은 사용자가 전달하는 화면 해상도 변수
    gl_FragColor = vec4(normalizedX, 0.0, 0.0, 1.0); // 좌에서 우로 선형 그라데이션하는 빨간색
}
```

