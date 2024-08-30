## Check the Ip of running containers

**inspect** 명령어를 활용하면 실행 중인 컨테이너의 IP 주소를 확인할 수 있습니다.

### windows

```bash
docker network inspect bridge | findstr "IPv4Address"
```

### linux

```bash
docker network inspect bridge | grep IPv4Address
```

