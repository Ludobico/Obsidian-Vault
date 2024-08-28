```dockerfile
FROM python:3.11.6

WORKDIR /usr/src/app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR ./myapp

CMD python manage.py runserver 0.0.0.0:8000
EXPOSE 8000

```

- **FROM** 명령어는 베이스 이미지를 의미합니다. 즉, 우리가 생성할 이미지 파일의 베이스가 python 3.11.6 이라는 의미입니다.

- **WORKDIR** 명령어는 리눅스의 **cd** 명령어와 비슷한 명령어로 해당 작업 디렉토리로 전환할 때 사용합니다. 즉, `/usr/src/app` 디렉토리로 전환하겠다는 의미입니다.

- **COPY** 명령어는 호스트에 존재하는 파일을 도커 이미지의 파일 시스템 경로로 복사하는 명령어입니다. 사용방법은 **COPY <호스트 파일 경로>:<이미지 파일 경로>** 로 **COPY . .** 명령어는 호스트의 현재 경로에서 디렉토리 내부에 존재하는 파일을 이미지 파일 경로의 현재 경로인 `/usr/src/app` 디렉토리로 복사한다는 의미입니다.

- **RUN** 명령어는 이미지 빌드 시 실행하고 싶은 명령어가 있을 때 사용하는 명령어입니다. 해당 명령어는 pip를 설치하는 명령어입니다.

- **CMD** 명령어로 서비스를 실행합니다.

>RUN 명령어는 이미지 빌드 시 실행되는 명령어를 입력하는 명령어이고 CMD 명령어는 컨테이너 실행 시 실행되는 명령어를 입력하는 명령어입니다.

- **EXPOSE** 명령어를 이용해 8000번 포트를 엽니다.

