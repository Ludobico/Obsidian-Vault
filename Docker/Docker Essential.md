- [[#Image management|Image management]]
		- [[#docker image ls|docker image ls]]
		- [[#docker pull \<image_name\>|docker pull \<image_name\>]]
		- [[#docker image rm \<image_id\>|docker image rm \<image_id\>]]
		- [[#docker build -t \<image_name\> .|docker build -t \<image_name\> .]]
- [[#Container management|Container management]]
		- [[#docker run \<image_name\>|docker run \<image_name\>]]
		- [[#docker run -d \<image_name\>|docker run -d \<image_name\>]]
		- [[#docker run -it \<image_name\> /bin/bash|docker run -it \<image_name\> /bin/bash]]
- [[#Container states|Container states]]
		- [[#docker container ls|docker container ls]]
		- [[#docker container ls -a|docker container ls -a]]
		- [[#docker container logs \<container_id\>|docker container logs \<container_id\>]]
- [[#Container control|Container control]]
		- [[#docker container stop \<container_id\>|docker container stop \<container_id\>]]
		- [[#docker container start \<container_id\>|docker container start \<container_id\>]]
		- [[#docker container rm \<container_id\>|docker container rm \<container_id\>]]
		- [[#docker container exec -it \<container_id\> \bin\bash|docker container exec -it \<container_id\> \bin\bash]]
		- [[#docker container exec \<container_id\> \<command\>|docker container exec \<container_id\> \<command\>]]
- [[#File management|File management]]
		- [[#docker cp \<host_path\> \<container_id\>:\<container_path\>|docker cp \<host_path\> \<container_id\>:\<container_path\>]]
		- [[#docker cp \<container_id\>:\<container_path\> \<host_path\>|docker cp \<container_id\>:\<container_path\> \<host_path\>]]
- [[#Resource management|Resource management]]
		- [[#docker system prune|docker system prune]]
		- [[#docker container prune|docker container prune]]

## Image management

#### docker image ls

- 로컬에 저장된 도커 이미지 목록을 확인

```shell
docker images
```


#### docker pull \<image_name\>

- Docker Hub에서 이미지를 다운로드

```
docker pull ubuntu:latest
```

#### docker image rm \<image_id\>

- 사용하지 않는 이미지를 삭제

```
docker image rm 7a8b9c3d2e1f
```

#### docker build -t \<image_name\> .

- 현재 디렉토리의 Dockerfile로 커스텀 이미지 생성

```
docker build -t my-app:1.0 .
```

## Container management

#### docker run \<image_name\>

- 이미지로 새 컨테이너 실행

```
docker run nginx
```

#### docker run -d \<image_name\>

- 백그라운드에서 컨테이너 실행

```
docker run -d nginx
```

#### docker run -it \<image_name\> /bin/bash

- 인터렉티브 모드로 컨테이너 실행 및 쉘 접속

```
docker run -it ubuntu /bin/bash
```

## Container states

#### docker container ls

- 실행 중인 컨테이너 목록 확인

```
docker container ls
```

#### docker container ls -a

- 모든 컨테이너 (종료된 것도 포함) 목록 확인

```
docker container ls -a
```

#### docker container logs \<container_id\>

- 컨테이너 로그 확인

```
docker container logs 4e5f6g7h8i9j
```

## Container control

#### docker container stop \<container_id\>

- 실행 중인 컨테이너 중지

```
docker container stop 4e5f6g7h8i9j
```

#### docker container start \<container_id\>

- 중지된 컨테이너 재시작

```
docker container start 4e5f6g7h8i9j
```

#### docker container rm \<container_id\>

- 컨테이너 삭제, 중지상태여야함

```
docker container rm 4e5f6g7h8i9j
```

#### docker container exec -it \<container_id\> \bin\bash

- 실행 중인 컨테이너에 쉘로 접속

```
docker container exec -it 4e5f6g7h8i9j /bin/bash
```

#### docker container exec \<container_id\> \<command\>

- 컨테이너 내에서 특정 명령 실행

```
docker container exec 4e5f6g7h8i9j ls
```

## File management

#### docker cp \<host_path\> \<container_id\>:\<container_path\>

호스트에서 컨테이너로 파일 복사

```
docker cp ./myfile.txt 4e5f6g7h8i9j:/app/
```

#### docker cp \<container_id\>:\<container_path\> \<host_path\>

컨테이너에서 호스트로 파일 복사

```
docker cp 4e5f6g7h8i9j:/app/myfile.txt ./
```

## Resource management

#### docker system prune

- 사용하지 않는 컨테이너, 이미지, 네트워크 삭제

```
docker system prune
```

#### docker container prune

- 종료된 컨테이너만 정리

```
docker container prune
```

