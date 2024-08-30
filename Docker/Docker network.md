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


## Create docker network

```bash
admin@BGR_AI G:\st002\Docker-test\ng-guni-djan\nginx>docker network ls
NETWORK ID     NAME      DRIVER    SCOPE
bf5c321d146a   bridge    bridge    local
1ce87da051da   host      host      local
0564c7a07af5   none      null      local

admin@BGR_AI G:\st002\Docker-test\ng-guni-djan\nginx>docker network create mynetwork02
0d282a9063c262273e2daf1c2699f08ba031654c9689735194e7e7d201d2c8f6

admin@BGR_AI G:\st002\Docker-test\ng-guni-djan\nginx>docker network ls
NETWORK ID     NAME          DRIVER    SCOPE
bf5c321d146a   bridge        bridge    local
1ce87da051da   host          host      local
0d282a9063c2   mynetwork02   bridge    local
0564c7a07af5   none          null      local
```

**docker network ls** 명령어로 네트워크 목록을 확인하고 **docker network create [네트워크이름]** 으로 네트워크가 생성된 것을 확인할 수 있습니다.

