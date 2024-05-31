`gsap.to()` 메서드는 **애니메이션이 시작할 때 요소의 현재 상태를 기준으로, 지정한 속성 값**으로 애니메이션을 만듭니다.

```js
gsap.to(targets, {vars});
```

- `targets` : 애니메이션을 대상 요소들입니다. DOM, 객체, 배열 등 다양한 형태가 가능합니다.
- `vars` : 애니메이션 속성 및 설정할 객체입니다. 여기에는 최종 상태, 지속 시간, 지연 시간, easeing, 콜백 등 다양한 설정이 포함됩니다.

```js
// 1초 동안, 요소의 x 위치를 100으로 이동
gsap.to(".my-element", { duration: 1, x: 100 });
```


파라미터로는 다음과 같습니다.

> callbackScope
- 모든 콜백(onStart, onUpdate, onComplete 등)에 사용할 범위(scope)를 지정합니다.
```js
gsap.to(".my-element", {
  x: 100,
  callbackScope: myObject,
  onUpdate: function() {
    console.log(this); // this는 myObject를 가리킵니다.
  }
});

```

> data
- 임의의 데이터를 할당할 수 있습니다. 이 데이터는 나중에 해당 트윈 인스턴스에서 참조할 수 있습니다.
```js
const myTween = gsap.to(".my-element", {
  x: 100,
  data: "exampleData"
});
console.log(myTween.data); // "exampleData"

```

> delay 
- 애니메이션이 시작되기 전에 기다리는 시간(초)입니다.
```js
gsap.to(".my-element", {
  x: 100,
  delay: 1 // 1초 후에 애니메이션 시작
});

```

> duration
- 애니메이션의 지속 시간(초)입니다. 기본값은 0.5입니다.
```js
gsap.to(".my-element", {
  x: 100,
  duration: 2 // 2초 동안 애니메이션
});

```

> ease
- 애니메이션의 속도 변화를 제어합니다. 예를 들어, "elastic" 또는 "strong.inOut" 등의 값을 사용할 수 있습니다.
```js
gsap.to(".my-element", {
  x: 100,
  ease: "power1.out"
});

```

> id
- gsap의 Tween 인스턴스에 고유 식별자를 할당할 수 있습니다. 이를 통해 나중에 `gsap.getById()` 로 해당 트윈을 찾을 수 있습니다.
```js
gsap.to(".my-element", {
  x: 100,
  id: "myTween"
});
// 나중에 트윈을 찾기
const myTween = gsap.getById("myTween");

```

> immediateRender
- 트윈이 인스턴스화되지마자 즉시 렌더링을 강제합니다. `to()` 트윈의 기본값은 `false` 이며, `from()` 및 `fromTo()` 트윈의 기본값은 `true` 입니다.

> inherit
- 트윈이 부모 타임라인의 기본 객체로부터 상속받을지 여부를 결정합니다. 기본값은 `true` 입니다.

> lazy 
- 트윈이 처음 렌더링될 때 시작 값을 읽고, 현재 tick의 끝까지 값을 쓰는 것을 지연시킵니다. **이는 성능을 향상시킬 수 있습니다.** , 기본값은 `true` 입니다.

> onComplete
- 애니메이션이 완료되었을 때 호출되는 함수입니다.

> onCompleteParams
- `onComplete` 함수에 전달할 매개변수 리스트입니다.
```js
function myFunction(param1, param2) {
  console.log(param1, param2);
}

gsap.to(".my-element", {
  x: 100,
  onComplete: myFunction,
  onCompleteParams: ["param1", "param2"]
});

```

> onInterrupt
- 트윈이 완료되기 전에 중단ㄴ될 때 호출되는 함수입니다. 이는 `kill()` 메서드가 호출되거나 덮어쓰기로 인해 발생할 수 있습니다.

> onInterruptParams
- `onInterrupt` 함수에 전달할 매개변수 리스트입니다.

> onRepeat
- 애니메이션이 반복될 때마다 호출되는 함수입니다. **바복이 설정된 경우에는 발생**합니다.

> onRepeatParams
- `onRepeat` 함수에 전달할 매개변수입니다.

> onReverseComplete
- 애니메이션이 반대 방향으로 재생되어 시작점이 도달했을 때 호출되는 함수입니다.

> onReverseCompleteParams
- `onReverseComplete` 함수에 전달할 매개변수 리스트입니다.

> onStart
- 애니메이션이 시작될 때 호출되는 함수입니다. 이는 애니메이션이 0에서 다른 값으로 변경될 때다 호출됩니다.

> onStartParams
- `onStart` 함수에 전달할 매개변수 배열입니다.

> onUpdate
- 애니메이션이 업데이트될 때마다 호출되는 함수입니다. 이는 애니메이션의 tick 마다 발생합니다.

> onUpdateParams
- `onUpdate` 함수에 전달할 매개변수 배열입니다.

> overwrite -> Default : false
- 트윈이 같은 대상에 적용된 다른 트윈에 덮어쓸지 여부를 결정합니다.
	-  `true` : 같은 대상의 모든 트윈을 즉시 제거합니다.
	-  `auto` : 처음 렌더링 시 충돌되는 부분만 제거합니다.
	-  `false` : 덮어쓰기를 사용하지 않습니다.

> paused -> Default : false
- `true` 로 설정하면, 애니메이션이 생성되자마자 일시 정지됩니다.

> repeat -> Default : 0
- 애니메이션이 반복될 횟수를 설정합니다. `repeat : 1` 이먄 총 두번 반복합니다. 무한반복은 `-1` 로 설정합니다.

> repeatDelay -> Default : 0
- 반복 사이의 대기 시간(초) 입니다.

> repeatRefresh -> Default : false
- `true`로 설정하면 반복할 때마다 시작/종료 값을 다시 기록합니다. 이는 동적 값을 사용할 때 유용합니다.

> reversed -> Default : false
- `true`로 설정하면 애니메이션이 거꾸로 시작됩니다.

> runBackwards -> Default : false
- `true`로 설정하면 시작과 종료 값이 반전됩니다. 이 속성은 `from()` 트윈과 동일하게 동작합니다.

> stagger
- 여러 대상의 시작 시간을 설정할 수 있습니다. 기본값은 시간 간격(초)이며, 객체를 사용하여 더 복잡한 설정도 가능합니다.
```js
gsap.to(".my-elements", {
  x: 100,
  stagger: 0.1
});

```

> startAt
- 애니메이션 시작 시 값을 설정합니다. 애니메이션되지 않는 속성도 설정할 수 있습니다.

> yoyo -> false
- `true` 로 설정하면 반복할 때마다 애니메이션이 반대 방향으로 실행됩니다.

> yoyoEase -> false
- yoyo 반복 단계에서 ease를 변경합니다. 특정 ease로 설정하거나 `true` 로 설정하며 일반 ease를 반전시킬 수 있습니다.

> keyframe
- 다양한 상태로 애니메이션하려면 `keyframe`을 사용합니다. 이는 `to()` 트윈의 배열로 작동합니다.
```js
gsap.to(".my-element", {
  keyframes: [
    { x: 100, duration: 1 },
    { y: 100, duration: 0.5 },
    { opacity: 0, duration: 0.5 }
  ]
});

```

