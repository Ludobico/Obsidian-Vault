
```
docker container ls
```

이 명령언느 현재 실행 중인 컨테이너 목록과 함께 외부에 포워딩된 포트를 보여줍니다.

```
CONTAINER ID   IMAGE         PORTS                    NAMES
abc123         nginx         0.0.0.0:8080->80/tcp     web_server
def456         redis         0.0.0.0:6379->6379/tcp   redis_server
```

## Instance Port

```
docker container ls --format "table {{.Names}}\t{{.Ports}}"
```

```docker
NAMES         PORTS
web_server    0.0.0.0:8080->80/tcp
redis_server  0.0.0.0:6379->6379/tcp
```

## Port of specific container

```
docker port <컨테이너 이름 or ID>
```

