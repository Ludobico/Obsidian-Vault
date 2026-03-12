OpenClaw는 모델에게 전달되는 메시지에 **timestamp envelope**을 추가하여  
모델이 **일관된 시간 기준(reference time)** 을 이해할 수 있도록 합니다.

```
[Provider ... 2026-01-05 16:26 PST] message text
```

이 timestamp의 **시간대, 표시 여부, 경과 시간 표시 여부**를 설정할 수 있습니다.

## 설정 위치

openclaw.json

```json
{
  "agents": {
    "defaults": {
      "envelopeTimezone": "local",
      "envelopeTimestamp": "on",
      "envelopeElapsed": "on"
    }
  }
}
```

## 설정 옵션

### envelopeTimezone

메시지 envelope에 표시되는 **시간대**를 설정합니다.

|값|설명|
|---|---|
|`local`|서버(OS)의 timezone 사용|
|`utc`|UTC 기준|
|`user`|`userTimezone` 설정 사용|
|`Asia/Seoul`|특정 timezone 직접 지정|

```json
{
  "agents": {
    "defaults": {
      "envelopeTimezone": "Asia/Seoul"
    }
  }
}
```

### envelopeTimestamp

타임스탬프 표시 여부

#### on

```
[Provider ... 2026-01-05 16:26 PST]
```

#### off

```
[Provider ...]
```

```json
"envelopeTimestamp": "off"
```


## Docker timezone과의 관계

OpenClaw의 timezone 설정은 **Docker OS timezone과 별개로 동작**합니다.

|구분|역할|
|---|---|
|Docker tzdata|컨테이너 OS 시간|
|envelopeTimezone|모델에게 보여주는 시간|
즉,

Docker timezone을 바꾸지 않아도 [[openclaw]] 에서 직접 timezone을 제어할 수 있습니다.

