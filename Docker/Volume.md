[[Docker]] 에서는 컨테이너의 데이터 저장을 관리하기 위해 여러 가지 스토리지 옵션을 제공합니다. 이 중 주요한 세 가지는 다음과 같습니다.

## 1. Volume

Docker 에서 가장 추천되는 스토리지 옵션입니다. 볼륨은 Docker가 관리하는 호스트 파일 시스템의 특정 위치에 데이터를 저장합니다.

## 2. Bind Mount

호스트 파일 시스템의 특정 디렉토리를 컨테이너에 연결하는 방식입니다. 컨테이너 내부에서 이 디렉토리를 마치 로컬 디렉토리처럼 사용할 수 있습니다.

## 3. Tmpfs Mount

컨테이너의 메모리에서만 데이터를 저장하는 방식입니다. 주로 데이터를 저장할 때 사용됩니다.

---

## Volume

이 중 **Volume** 방식을 활용해 데이터를 관리해보겠습니다. volume은 <font color="#ffff00">도커 컨테이너에서 생성되는 데이터가 컨테이너를 삭제한 후에도 유지될 수 있도록 도와주는 저장소</font>입니다. volume은 도커에 의해 관리됩니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo>docker volume ls
DRIVER    VOLUME NAME
local     91a5264007d4c15ed1391bd86e38713d33e512fc84af375d3bd9f4fb3bc1269e
local     40138cfb29fb009ddfaef5781ef0fd93588659b10a98300fb9070c9ca6624e1c

admin@BGR_AI C:\Users\admin\Desktop\repo>docker volume create myvolume01
myvolume01

admin@BGR_AI C:\Users\admin\Desktop\repo>docker volume ls
DRIVER    VOLUME NAME
local     91a5264007d4c15ed1391bd86e38713d33e512fc84af375d3bd9f4fb3bc1269e
local     40138cfb29fb009ddfaef5781ef0fd93588659b10a98300fb9070c9ca6624e1c
local     myvolume01
```

- **docker volume ls** 명령어를 입력하면 도커 볼륨 리스트를 확인할 수 있습니다.

- **docker volume create [도커 볼륨명]** 명령어를 입력하면 자신이 원하는 도커 볼륨을 생성할 수 있습니다. 

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo>docker run -e POSTGRES_PASSWORD=mysecretpassword --mount type=volume,source=myvolume01,target=/var/lib/postgresql/data -d postgres
93077aa35211f88d584112e4cba0152e06e5edcf1774dbc42d00f19f5b0c75c7
```

- 도커 볼륨 `myvolume01` 과 연동시켜 PostgreSQL 컨테이너로 실행하겠습니다. **--mount** 옵션을 활용해 **source=[도커 볼륨명],target=[컨테이너 내부 경로]** 형태로 사용합니다. 명령어중 쉼표(,)를 사용할 때 띄워쓰기를 하지 않는다는 점에 주의해야 합니다. <font color="#ffff00">위 과정은 myvolume01 볼륨과 컨테너 내부의 /var/lib/postgresql/data 경로를 연결시키는 것을 의미</font>합니다.

이렇게 컨테이너를 실행하면 컨테이너 내부의 /var/lib/postgresql/data 경로에 존재하는 파일은 모두 도커 볼륨에 보관됩니다. 참고로 /var/lib/postgresql/data 은 PostgreSQL 에서 데이터가 보관되는 경로입니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo>docker container ls
CONTAINER ID   IMAGE      COMMAND                   CREATED         STATUS         PORTS      NAMES
93077aa35211   postgres   "docker-entrypoint.s…"   4 minutes ago   Up 4 minutes   5432/tcp   sad_shannon

admin@BGR_AI C:\Users\admin\Desktop\repo>docker exec -it 93077aa35211 /bin/bash
root@93077aa35211:/# psql -U postgres
psql (16.4 (Debian 16.4-1.pgdg120+1))
Type "help" for help.

postgres=# CREATE USER user01 PASSWORD '1234' SUPERUSER;
CREATE ROLE
postgres=# \du
                             List of roles
 Role name |                         Attributes
-----------+------------------------------------------------------------
 postgres  | Superuser, Create role, Create DB, Replication, Bypass RLS
 user01    | Superuser

postgres=# \q
root@93077aa35211:/# cd /var/lib/postgresql/data/
root@93077aa35211:/var/lib/postgresql/data# ls
base    pg_commit_ts  pg_hba.conf    pg_logical    pg_notify    pg_serial     pg_stat      pg_subtrans  pg_twophase  pg_wal   postgresql.auto.conf  postmaster.opts
global  pg_dynshmem   pg_ident.conf  pg_multixact  pg_replslot  pg_snapshots  pg_stat_tmp  pg_tblspc    PG_VERSION   pg_xact  postgresql.conf       postmaster.pid
root@93077aa35211:/var/lib/postgresql/data# exit
exit
```

- 컨테이너 내부에 접속합니다.
- **psql** 명령어를 입력해서 postgres 사용자로 PostgreSQL에 접속합니다.
- user01 이라는 사용자를 SUPERUSER 권한으로 생성합니다.
- 사용자 목록을 확인하면 방금 생성한 user01 이라는 사용자를 확인할 수 있습니다. 바로 이 user01 이라는 사용자 데이터가 도커 볼륨에 의해 유지될 예정입니다.
- PostgreSQL을 종료합니다.
- /var/lib/postgresql/data 로 이동합니다.
- 파일 목록을 확인합니다. **ls** 명령어로 나온 파일들이 도커 볼륨에 저장될 예정입니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo>docker container ls
CONTAINER ID   IMAGE      COMMAND                   CREATED         STATUS         PORTS      NAMES
93077aa35211   postgres   "docker-entrypoint.s…"   9 minutes ago   Up 9 minutes   5432/tcp   sad_shannon

admin@BGR_AI C:\Users\admin\Desktop\repo>docker container stop 93077aa35211
93077aa35211

admin@BGR_AI C:\Users\admin\Desktop\repo>docker container rm 93077aa35211
93077aa35211
```

- 실행중인 컨테이너를 정지시키고 삭제합니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo>docker image ls  
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
my-ubuntu    0.1       985949ed7822   5 hours ago     119MB
postgres     latest    69092dbdec0d   2 weeks ago     432MB
ubuntu       latest    edbfe74c41f8   3 weeks ago     78.1MB
python       3.11.6    0dba5a08d425   10 months ago   1.01GB

admin@BGR_AI C:\Users\admin\Desktop\repo>docker container run -e POSTGRES_PASSWORD=mysecretpassword -v myvolume01:/var/lib/postgresql/data -d postgres
7494b68df0c9dab8b2f4f1b6b33ea9bf8a818ada58dbb61c197cb3b87d7870af

admin@BGR_AI C:\Users\admin\Desktop\repo>docker container ls
CONTAINER ID   IMAGE      COMMAND                   CREATED         STATUS         PORTS      NAMES
7494b68df0c9   postgres   "docker-entrypoint.s…"   4 seconds ago   Up 3 seconds   5432/tcp   elastic_diffie

admin@BGR_AI C:\Users\admin\Desktop\repo>docker exec -it 7494b68df0c9 /bin/bash
root@7494b68df0c9:/# psql -U postgres
psql (16.4 (Debian 16.4-1.pgdg120+1))
Type "help" for help.

postgres=# \du
                             List of roles
 Role name |                         Attributes
-----------+------------------------------------------------------------
 postgres  | Superuser, Create role, Create DB, Replication, Bypass RLS
 user01    | Superuser

postgres=# \q
root@7494b68df0c9:/# exit
exit
```

- postgres 이미지를 이용해 PostgreSQL 컨테이너를 실행합니다. 앞선 실습과 마찬가지로 `myvolume01` 볼륨과 컨테이너 내부 경로 /var/lib/postgresql/data 를 연결해서 실행합니다. 도커 볼륨을 컨테이너 내부 경로에 사용할때는 앞선 실습과 같이 **--mount** 옵션을 사용할 수도 있고 이번처럼 **-v** 옵션을 사용할 수도 있습니다. -v에서 v는 volume의 줄임말로 **--volume** 형태로 사용할 수도 있습니다.
- 사용자 목록을 확인하면 이전 실습에서 생성했던 user01이 존재하는 것을 볼 수 있습니다. 이처럼 도커볼륨을 활용하면 컨테이너가 삭제되어도 컨테이너 내부 데이터를 관리하기 편하다는 것을 알 수 있습니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo>docker volume inspect myvolume01
[
    {
        "CreatedAt": "2024-08-28T02:29:50Z",
        "Driver": "local",
        "Labels": null,
        "Mountpoint": "/var/lib/docker/volumes/myvolume01/_data",
        "Name": "myvolume01",
        "Options": null,
        "Scope": "local"
    }
]
```

- 도커 호스트에서 **inspect** 명령어를 사용하면 볼륨의 정보를 확인할 수 있습니다.
- `Mountpoint` 가 컨테이너의 데이터를 보관하는 로컬 호스트 경로입니다. 즉, myvolume01이라는 볼륨에서 관리하는 데이터가 존재하는 경로를 /var/lib/docker/volumes/myvolume01/\_data 라는 뜻입니다.


## Bind mount

두 번째로는 도커 스토리지 종류 중 하나인 `bind mount` 에 대해 알아보겠습니다. bind mount 방식은 <font color="#ffff00">도커 호스트의 디렉토리와 컨테이너 디렉토리를 연결시켜 데이터를 보관하는 방식</font>입니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo\bind-mount>dir
 C 드라이브의 볼륨에는 이름이 없습니다.
 볼륨 일련 번호: 8219-0BCE

 C:\Users\admin\Desktop\repo\bind-mount 디렉터리

2024-08-28  오후 02:02    <DIR>          .
2024-08-28  오후 02:02    <DIR>          ..
2024-08-28  오후 02:02                 0 test01.txt
2024-08-28  오후 02:02                 0 test02.txt
               2개 파일                   0 바이트
               2개 디렉터리  280,560,840,704 바이트 남음
```

![[Pasted image 20240828140651.png]]


- 해당 디렉토리에 존재하는 파일 (test01, test02) 파일을 컨테이너에 유지시켜 보겠습니다. 위 파일은 도커호스트뿐만아니라 컨테이너에도 저장될 예정입니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop\repo\bind-mount>docker container run -e POSTGRES_PASSWORD=mysecretpassword --mount type=bind,source=C:\Users\admin\Desktop\repo\bind-mount,target=/work -d postgres
7c4815e7559d0be7796b8091876a228ee171f4498405f56ea502a34e9583bf8c

admin@BGR_AI C:\Users\admin\Desktop\repo\bind-mount>docker container ls
CONTAINER ID   IMAGE      COMMAND                   CREATED              STATUS              PORTS      NAMES
7c4815e7559d   postgres   "docker-entrypoint.s…"   About a minute ago   Up About a minute   5432/tcp   hardcore_curie
7494b68df0c9   postgres   "docker-entrypoint.s…"   22 minutes ago       Up 22 minutes       5432/tcp   elastic_diffie

admin@BGR_AI C:\Users\admin\Desktop\repo\bind-mount>docker container exec -it 7c4815e7559d /bin/bash
root@7c4815e7559d:/# ls
bin  boot  dev  docker-entrypoint-initdb.d  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var  work
root@7c4815e7559d:/# cd work/
root@7c4815e7559d:/work# ls
test01.txt  test02.txt
```

- 위와 같이 컨테이너를 실행할 때 도커 볼륨을 사용하기 위해 **--mount** 옵션을 사용하고 도커 호트의 `C:\Users\admin\Desktop\repo\bind-mount` 경로와 도커 컨테이너 내부의 `/work` 경로를 연결시켜줍니다. 
- work 디렉토리를 확인해보면 도커 호스트에 존재하는 test01 파일과 test02 파일이 존재하는 것을 알 수 있습니다.

```bash
root@7c4815e7559d:/work# mkdir test_dir
root@7c4815e7559d:/work# ls
test01.txt  test02.txt  test_dir
```

- 해당 경로의 파일 목록을 확인해보면 컨테이너 내부에서 생성했던 `test_dir` 디렉토리가 도커호트에도 생성되어 있는 것을 알 수 있습니다. 즉, 컨테이너 내부에서 파일이 변하면 연결되어 있는 도커 호스트 경로도 함께 변하는 것을 알 수 있습니다.

![[Pasted image 20240828141204.png]]

