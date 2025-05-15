- [[#Install release build|Install release build]]
- [[#Setting ADB|Setting ADB]]
	- [[#Setting ADB#Windows|Windows]]
	- [[#Setting ADB#Linux / Mac|Linux / Mac]]
	- [[#Setting ADB#Setting Path|Setting Path]]
- [[#Encoding Issue on CMD|Encoding Issue on CMD]]
- [[#Check Logs|Check Logs]]


[[React Native]] 프로젝트에서 AVD로 릴리스 빌드를 설치하고, `console.log` 를 포함한 로그를 확인하는 방법을 정리합니다. **릴리스 모드에서는 디버그 모드와 달리 로그 확인이 까다로울 수 있으므로** adb와 인코딩 설정을 활용한 방법을 다룹니다.
## Install release build

릴리스 빌드를 AVD에 설치하려면 다음 명령어를 사용합니다.

```
cd android
```

```
gradlew installRelease
```

## Setting ADB

<font color="#ffff00">adb(Android Debug Bridge)</font> 는 **AVD와 통신하여 로그를 확인하는 데 사용되는 도구**입니다. ADB 실행 파일은 Android SDK의 <font color="#ffff00">platform-tools</font> 폴더에 있습니다.

### Windows

```
C:\Users\<사용자이름>\AppData\Local\Android\Sdk\platform-tools
```

### Linux / Mac

```
~/Library/Android/sdk/platform-tools
```

### Setting Path

ADB를 명령어로 쉽게 실행하려면, <font color="#ffff00">platform-tools</font> 경로를 시스템 PATH에 추가하세요.

- `제어판 > 시스템 > 고급 시스템 설정 > 환경 변수` 로 이동.
- `Path` 변수에 경로 추가

설정 후 CMD 에서 `adb --version` 을 실행해 설치 확인

## Encoding Issue on CMD

릴리스 빌드에서 `console.log` 를 확인하려면 `adb logcat` 을 사용하지만, **윈도우 CMD 에서 한글 인코딩 문제로 로그가 깨기저나 필터링이 실패**할 수 있습니다.

- CMD는 기본적으로 EUC-KR(코드 949)를 사용합니다.
- 소스 코드(예: a.py)나 로그 파일이 **UTF-8**로 작성된 경우, CMD에서 한글이 깨져 보이거나 findstr로 검색이 실패할 수 있습니다.
- 예: findstr /si /c"문자" \*.py로 검색 시, UTF-8 파일에서 한글을 찾지 못함.

CMD에서 인코딩을 UTF-8로 변경하려면 `chcp` 명령어를 사용합니다.

```
chcp
```

- 윈도우 기준, 기본값은 949 (EUC-KR) 입니다.

```
chcp 65001
```

- 코드 페이지가 65001 (utf-8)로 변경됩니다.

## Check Logs

릴리스 빌드에서 `console.log` 를 확인하려면 `adb logcat` 과 `findstr` 를 조합합니다.

```
adb logcat | findstr ReactNativeJS
```

