
Windows 환경에서 `React Native Reanimated` 라이브러리를 설치하고 빌드할 때 다음과 같은 에러가 발생할 수 있습니다.

## Build command

```bash
npx expo prebuild -p android
```

or

```bash
npx expo prebuild -p android --clean
```


## Error

```bash
> Task :react-native-reanimated:buildCMakeDebug[arm64-v8a] FAILED

FAILURE: Build failed with an exception.

* What went wrong:
Execution failed for task ':react-native-reanimated:buildCMakeDebug[arm64-v8a]'.

...
No such file or directory
```

이는 Windows의 파일 경로 길이 제한으로 인한 문제입니다. 해결을 위해서는 cmake 폴더 내의 기존 **ninja 파일을 최신 버전으로 교체**해야 합니다


- [관련 깃허브 이슈](https://github.com/software-mansion/react-native-reanimated/issues/6872)

- [ninja relase](https://github.com/ninja-build/ninja/releases)

윈도우 최신 버전의 ninja를 다운받고 아래의 경로로 접근합니다.

```
C:\Users\{사용자이름}\AppData\Local\Android\Sdk\cmake\VERSION\bin
```

해당폴더의 `ninja.exe` 파일을 최신 버전으로 교체한 뒤 커맨드 추가 명령어를 입력합니다.

```
yarn cache clean
```

```
cd android
gradlew clean
```

그 뒤, 다시 리빌드합니다.

```
yarn android
```

