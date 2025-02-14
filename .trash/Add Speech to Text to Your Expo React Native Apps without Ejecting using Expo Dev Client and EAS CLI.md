
https://www.youtube.com/watch?v=gpXF9heaw8k

## OverView

[[React Native]] Expo 환경에서 STT(Speech-To-Text) 기능을 구현하기 위해서는 다음 두 가지 방법 중 하나를 선택해야 합니다.

```
Expo Dev Client 와 EAS CLI를 사용하여 개발 빌드된 앱으로 실행
```

```
Expo Bare Workflow 로 전환하여 개발
```

여기에서는 첫 번째 방법으로 개발합니다.

빌드에 관련된 세팅과 주의사항은 [[Build react native expo]] 항목을 참조해주세요.

## Command

라이브러리를 설치합니다.

```
yarn add expo-dev-client expo-speech-recognition
```

`app.json` 파일에 다음 구문을 추가로 작성합니다.

```JSON
"plugins": [
  [
    "@react-native-voice/voice",
    {
      "microphonePermission": "Allow Voice to Text Tutorial to access the microphone",
      "speechRecognitionPermission": "Allow Voice to Text Tutorial to securely recognize user speech"
    }
  ]
]
```

#### optional
`EAS-CLI` 가 설치되어 있지 않다면 아래의 커맨드로 설치합니다.

```bash
npm install -g eas-cli
```



아래의 명령어로 로그인 및 빌드합니다.

```
eas login
eas build:configure
```

아래의 명령어로 expo-dev-client 로 빌드합니다.

#### windows
```
eas build -p ios --profile development
OR
eas build -p android --profile development
```

#### linux or macOS
```
eas build -p ios --profile development --local
OR
eas build -p android --profile development --local
```

아래의 명령어로 실행합니다.

```
expo start --dev-client
```

