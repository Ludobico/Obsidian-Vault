<font color="#ffff00">도커 파일(Dockerfile)</font> 은 **사용자가 명령 줄에서 호출할 수 있는 모든 명령을 포함하는 텍스트 문서**입니다. 이 파일은 도커 이미지를 조립하기 위한 명령어들을 순차적으로 기술합니다. 여기에는 Docker가 이미지를 빌드할 때 자동으로 읽는 명령어들이 포함됩니다.

## Dockerfile command
---

- <font color="#ffff00">LABEL</font> : 이미지에 메타데이터를 추가합니다. 주로 **이미지의 버전,제작자,라이선스 등을 포함**합니다.

```dockerfile
LABEL key = value
```

- <font color="#ffff00">WORKDIR</font> : 생성된 컨테이너 내에서 명령어를 실행할 디렉터리를 설정합니다.

```dockerfile
WORKDIR /path/to/directory
```

- <font color="#ffff00">COPY</font> : 호스트 파일이나 디렉토리를 이미지로 복사하여 추가합니다.

```dockerfile
COPY source destination
```

- <font color="#ffff00">ADD</font> : 호스트의 파일이나 디렉토리를 이미지로 추가합니다. COPY와는 다르게 압축 파일을 자동으로 해제하고 추가할 수 있으며, 원격에서 파일을 다운로드하여 추가하는 기능도 제공합니다. 주로 COPY보다는 더 많은 기능을 제공하는 경우에 사용합니다.

```DOCKERFILE
ADD source destination
```

- <font color="#ffff00">EXPOSE</font> : 컨테이너와 호스트 사이의 포트를 설정합니다. 이 명령은 컨테이너에서 리스닝하는 를 지정할 뿐, 호스트와의 연결은 설정하지 않습니다. 따라서 일반적으로 `docker run -p` 명령을 사용하여 호스트와의 연결을 설정합니다.

```dockerfile
EXPOSE port
```

- <font color="#ffff00">ENV</font> : 환경 변수를 설정합니다. 도커 이미니 내에서 사용할 환경 변수를 정의합니다.

```dockerfile
ENV key=value
```

- <font color="#ffff00">RUN</font> : 이미지를 만들기 위해 컨테이너 내부에서 실행할 명령어를 지정합니다. 주로 **패키지 설치, 빌드** 작업 등을 수행합니다.

```dockerfile
RUN command
```

- <font color="#ffff00">CMD</font> : 컨테이너가 실행될 때 바로 실행할 명령어를 설정합니다. Dockerfile 내에서 한 번만 사용할 수 있으며, 여러 번 사용한 경우 마지막 명령어만 실행됩니다.

```dockerfile
CMD ["executable", "param1", "param2"]
```

- <font color="#ffff00">ENTRYPOINT</font> : 컨테이너가 시작될 때 실행할 스크립트나 실행 파일을 설정합니다. 일반적으로 CMD와 함께 사용되며, CMD의 인자로 사용됩니다.

```dockerfile
ENTRYPOINT ["executable", "param1", "param2"]
```

- <font color="#ffff00">VOLUME</font> : 호스트와 컨테이너 간에 데이터를 공유하기 위한 볼륨을 마운트합니다. 주로 데이터베이스 파일이나 로그 파일과 같이 **영구적인 데이터를 저장하기 위해 사용**됩니다.

```dockerfile
VOLUME /path/to/directory
```

