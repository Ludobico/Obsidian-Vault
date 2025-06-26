- [[#Installation|Installation]]
	- [[#Installation#Linux|Linux]]
	- [[#Installation#Windows|Windows]]
- [[#Configuration of docker-compose|Configuration of docker-compose]]
- [[#Volumes vs COPY|Volumes vs COPY]]
	- [[#Volumes vs COPY#volumes|volumes]]
	- [[#Volumes vs COPY#COPY|COPY]]
- [[#compose up --build  -d vs compose up -d|compose up --build  -d vs compose up -d]]


도커 컴포즈(docker compose) 는 도커를 활용해 <font color="#ffff00">다수의 컨테이너 형태의 애플리케이션을 실행할 수 있는 도구</font>입니다. 실행하고자 하는 애플리케이션의 설정 내용들을 `YAML` 파일로 작성하는 방법으로 도커 컴포즈를 활용할 수 있습니다. YAML 파일 작성을 완료하면 간단한 명령어만으로도 YAML에 포함되어 있는 모든 서비스를 한번에 실행할 수 있습니다.

## Installation

### Linux

```bash
sudo pip3 install docker-compose
docker compose version
```

### Windows

윈도우의 경우 Docker Desktop을 설치하면 도커 컴포즈도 함께 설치됩니다.

```bash
docker compose version
```

## Configuration of docker-compose

```bash
version: '3'

services:
  djangotest:
    build: ./myDjango03
    networks:
      - composenet01
    depends_on:
      - postgrestest
    restart : always

  nginxtest:
    build: ./myNginx03
    networks:
      - composenet01
    ports:
      - "6966:80"
    depends_on:
      - djangotest
    restart : always
  
  postgrestest:
    build: ./myPostgres03
    networks:
      - composenet01
    environment:
      POSTGRES_USER : postgres
      POSTGRES_PASSWORD : mysecretpassword
      POSTGRES_DB : postgres
    volumes:
      - composevol01:/var/lib/postgresql/data
    restart: always

networks:
  composenet01:

volumes:
  composevol01:
```

```bash
docker compose up -d
```

```
 => [postgrestest internal] load build definition from Dockerfile                                                                                           0.1s
 => => transferring dockerfile: 57B                                                                                                                         0.0s 
 => [postgrestest internal] load metadata for docker.io/library/postgres:latest                                                                             0.0s 
 => [postgrestest internal] load .dockerignore                                                                                                              0.1s
 ...

[+] Running 5/5
 ✔ Network docker-compose_composenet01      Created                                                                                                         0.2s 
 ✔ Volume "docker-compose_composevol01"     Created                                                                                                         0.0s 
 ✔ Container docker-compose-postgrestest-1  Started                                                                                                         1.9s 
 ✔ Container docker-compose-djangotest-1    Started                                                                                                         2.4s 
 ✔ Container docker-compose-nginxtest-1     Started   
```

각각의 코드를 해석하면 다음과 같습니다.

```yaml
version: '3'
```

컴포즈 파일 포맷 버전 정보를 입력합니다. 컴포즈 파일 포맷 버전은 크게 1 버전, 2 버전, 3 버전으로 나뉘는데 현재는 3버전을 사용합니다.

```yaml
services:
```

실행하고자 하는 서비스 목록을 입력합니다.

```yaml
djangotest:
```

django를 활용한 서비스 이름을 djangotest라고 지었습니다. 이는 곳 이미지의 이름이 됩니다.

```yaml
build: ./myDjango03
```

이미지를 빌드할 디렉토리 경로를 적어줍니다.

```yaml
    networks:
      - composenet01
```

해당 서비스가 사용할 도커 네트워크 정보를 입력합니다.


```yaml
    depends_on:
      - postgrestest
```

`depends_on` 은 <font color="#ffff00">컨테이너 실행 순서</font>를 정할 때 사용됩니다. 만약 postgrestest가 입력되어있다면 postgrestest 컨테이너를 먼저 실행한 후 djangotest가 나중에 실행되는 것입니다.

```yaml
    restart : always
```

**restart : always** 는 컨테이너가 정지되면 재실행하라는 명령어입니다.

```yaml
  nginxtest:
    build: ./myNginx03
    networks:
      - composenet01
```

Nginx 서비스에 관한 정보입니다. 기본적으로 빌드하고자하는 이미지 경로를 입력하고 사용할 [[Docker network]] 정보를 입력합니다.

```yaml
    ports:
      - "6966:80"
```

**<도커 호스트 포트>:<컨테이너 포트>** 형태로 포트포워딩 정보를 입력합니다. - "6966:80" 에서 앞의 6966은 도커 호스트 포트를 의미하고 80은 도커 컨테이너 포트를 의미합니다.

```yaml
    depends_on:
      - djangotest
```

`depends_on` 을 통해 djangotest가 먼저 실행된 후에 nginxtest가 실행되도록 설정합니다.

```yaml
  postgrestest:
    build: ./myPostgres03
    networks:
      - composenet01
```

PostgreSQL 서비스에 관한 정보입니다. 기본적으로 빌드하고자하는 이미지 경로를 입력하고 도커네트워크 정보를 입력합니다.

```yaml
    environment:
      POSTGRES_USER : postgres
      POSTGRES_PASSWORD : mysecretpassword
      POSTGRES_DB : postgres
```

PostgreSQL 컨테이너에 포함될 환경 변수 정보를 입력합니다.

```yaml
    volumes:
      - composevol01:/var/lib/postgresql/data
```

PostgreSQL 이 [[Docker Volume]] 을 사용하도록 설정합니다. 따라서 `volumes` 를 통해 도커 볼륨 정보를 입력합니다. `composevol01:/var/lib/postgresql/data` 는 composevol01 이라는 볼륨을 PostgreSQL 컨테이너 내부의 `/var/lib/postgresql/data` 경로에 마운트하겠다는 의미입니다.

```yaml
networks:
  composenet01:
```

네트워크 정보를 입력합니다.

```yaml
volumes:
  composevol01:
```

볼륨 정보를 입력합니다.

```bash
admin@BGR_AI G:\st002\Docker-test\docker-compose>docker compose down
time="2024-09-02T15:37:40+09:00" level=warning msg="G:\\st002\\Docker-test\\docker-compose\\docker-compose.yaml: `version` is obsolete"
[+] Running 4/4
 ✔ Container docker-compose-nginxtest-1     Removed                                                                                                         0.8s 
 ✔ Container docker-compose-djangotest-1    Removed                                                                                                        10.7s 
 ✔ Container docker-compose-postgrestest-1  Removed                                                                                                         0.8s 
 ✔ Network docker-compose_composenet01      Removed                                                                                                         0.4s 
```

*docker compose down* 명령어를 활용하면 도커 컴포즈를 활용해 실행했던 컨테이너를 정지시킬 수 있습니다.

## Volumes vs COPY

`volumes` 옵션과 `COPY` 명령어는 [[Docker]]에서 파일을 컨테이너에 복사하는 방식에 있어서 차이점이 있습니다.

### volumes

- `volumes` 는 호스트 시스템의 파일이나 디렉터리를 컨테이너 내의 특정 경로에 **마운트하는 방식**입니다.

- 이 방식은 **호스트 시스템에서 파일을 수정하면, 수정된 내용이 컨테이너 내에서도 반영**됩니다.

```dockerfile
volumes:
  - ./nginx/default.conf:/etc/nginx/conf.d/default.conf
```

여기서는 호스트의 `./nginx/default.conf` 파일을 컨테이너의 `/etc/nginx/conf.d/default.conf` 경로에 마운트하게 됩니다. 호스트에서 해당 파일을 수정하면 컨테이너 안에서도 즉시 반영됩니다.

### COPY

- `COPY` 는 빌드할 때 호스트 시스템의 파일을 컨테이너 이미지 안에 복사하는 명령어입니다.

- 이 방식은 도커 이미지가 빌드될 때 파일을 컨테이너로 복사하며, 이미지가 빌드된 후에는 호스의 파일과는 연결되지 않습니다. 즉, 파일을 수정하면 컨테이너에는 반영되지 않으며, 이미지를 새로 빌드해야 합니다.


## compose up --build  -d vs compose up -d

`docker compose up --build -d` 와 `docker compose up -d`는 비슷해 보이지만, **빌드 단계의 차이**가 명확합니다.

|명령어|설명|
|---|---|
|`docker compose up -d`|기존에 **이미 빌드된 이미지가 있으면 그것을 사용**해서 컨테이너 실행|
|`docker compose up --build -d`|먼저 **이미지를 새로 빌드한 뒤**, 그 이미지로 컨테이너 실행|
🔸 docker compose up -d
Dockerfile이나 코드에 변경 사항이 없을 때

컨테이너만 다시 띄우고 싶을 때

🔸 docker compose up --build -d
Dockerfile이나 COPY한 파일들에 변경이 있었을 때
예: nginx 설정(default.conf), FastAPI 코드, requirements.txt 등

이미지를 다시 만들어야 할 때 반드시 필요

|명령어|설명|
|---|---|
|`docker compose build`|이미지 빌드만 수행 (컨테이너 실행 안 함)|
|`docker compose up -d`|기존 이미지로 컨테이너만 실행|
|`docker compose up --build -d`|이미지 새로 빌드하고 컨테이너 실행|
|`docker compose down`|모든 컨테이너 종료 및 삭제 (volume 제외)|
