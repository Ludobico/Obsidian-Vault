---
{}
---
Three.js 는 Javascript에서 3D 그래픽을 다루기 위한 라이브러리이며, Hooks는 React Three Fiber(R3F) 라이브러리에 종속됩니다. 

**Hooks는 컨텍스트에 의존하기 때문에 캔버스 요소 내에서만 사용할 수 있습니다**

## bad example
```js
import { useThree } from '@react-three/fiber'

function App() {
  const { size } = useThree() // This will just crash
  return (
    <Canvas>
      <mesh>
```

## good example

```js
function Foo() {
  const { size } = useThree()
  ...
}

function App() {
  return (
    <Canvas>
      <Foo />
```