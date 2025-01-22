- [[#Step1. Sing up to Expo|Step1. Sing up to Expo]]
- [[#Step2. Install expo package|Step2. Install expo package]]
- [[#Step3. Create expo project|Step3. Create expo project]]
- [[#Step4. Deploy to Expo|Step4. Deploy to Expo]]
- [[#Caution|Caution]]

## Step1. Sing up to Expo

![[Pasted image 20240902131532.png]]

expo 회원가입 페이지 접속 및 회원가입 진행

## Step2. Install expo package

![[Pasted image 20240902131601.png]]

VS코드에서 빌드할 프로젝트 폴더의 터미널 열기

```bash
npm install -g eas-cli
```

패키지 설치 명령어 실행

```bash
npx create-expo-app [App-Name]
```

프로젝트 생성 명령어로 expo 프로젝트 생성

```bash
cd [App-Name] 
npx expo login
```

생성한 프로젝트로 이동 및 expo 로그인 정보 등록

![[Pasted image 20240902131728.png]]

이메일 및 비밀번호 입력

## Step3. Create expo project

![[Pasted image 20240902131754.png]]

Expo 로그인 및 All project -> Create Projet 클릭

![[Pasted image 20240902131822.png]]

Display Name 입력 -> Create 클릭

![[Pasted image 20240902131842.png|512]]

id 명령어 복사

![[Pasted image 20240902131900.png]]

프로젝트 ID 적용

## Step4. Deploy to Expo

```bash
eas build -p android --profile preview
```

빌드 명령어 입력

![[Pasted image 20240902132037.png]]

애플리케이션 id 입력 (수정하지 않으면 프로젝트 이름인 기본 값으로 진행) → enter

Key 스토어 생성 여부 -> Y or N

이후 10분 후 베포 완료 (무료 사용자의 베포는 ‘사용자 대기 시스템’으로 진행)


![[Pasted image 20240902132056.png]]

## Caution

Expo로 배포한 react native는 **기본적으로 http 통신이 불가능**하게 되어있으므로, 추가적인 패키지 설치가 필요

```bash
npx expo install expo-build-properties
```

패키지 설치 후 해당 프로젝트의 `app.json` 에 추가적인 플러그인 입력이 필요

```json
    "plugins": [
      [
        "expo-build-properties",
        {
          "android": {
            "usesCleartextTraffic": true
          }
        }
      ]
    ]
```

