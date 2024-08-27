
## image list

**docker image ls** 명령어로 다운로드한 이미지 목록을 확인할 수 있습니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker image ls
REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
ubuntu        latest    edbfe74c41f8   3 weeks ago     78.1MB
python        3.11.6    0dba5a08d425   10 months ago   1.01GB
hello-world   latest    d2c94e258dcb   16 months ago   13.3kB
```

- REPOSITORY는 이미지 이름을 의미하며 TAG 는 이미지 태그를 의미합니다. IMAGE ID는 다운로한 이미지와 ID를 나타내는데, 이때 IMAGE ID는 다운로드할 때의 DIGEST 값과 다르다는 것을 알 수 있습니다. 다운로드할 때의 DIGEST 값은 도커 레지스트리에 존재하는 이미지의 DIGEST 값이고 `docker image ls`  의 결과값으로 나오는 IMAGE ID 값은 **다운로드한 후에 로컬에서 할당받은 IMAGE ID 값에 해당**합니다. CREATED는 이미지가 만들어진 시간을 의미하며, SIZE는 이미지 크를 나타냅니다.

## Create and execute docker container

```bash

admin@BGR_AI C:\Users\admin\Desktop>docker container run ubuntu

admin@BGR_AI C:\Users\admin\Desktop>

```

우분투 이미지를 컨테이너로 실행했습니다.

- 앞서 hello world를 실행할 때는 단순히 **docker run [이미지 이름]** 이라는 명령어를 입력했는데, 이번에는 **docker container run [이미지 이름]** 명령어를 입력했습니다. docker run 명렁어는 도커 초기 버전에서 활용한 명령어인데 현재는 **docker container run** 명령어가 권장됩니다.

- 코드 실행 결과 아웃풋이 따로 출력되지는 않는 것을 알 수 있습니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container run python:3.11.6

admin@BGR_AI C:\Users\admin\Desktop>
```

이번에는 [[Python]]3.11.6 이미지를 컨테이너로 실행하겠습니다. 위와 같이 명령어를 입력하면 앞선 실습과 마찬가지로 아웃풋은 따로 출력되지 않는 것을 알 수 있습니다.

## docker container list

**docker container ls**를 입력하면 도커 컨테이너 목록을 확인할 수 있습니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container ls
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES

admin@BGR_AI C:\Users\admin\Desktop>
```

위 결과를 확인해보면 컨테이너가 출력되지 않는 것을 알 수 있습니다. 그 이유는 별다른 옵션을 지정하지 않고 기본 형태로 docker container ls 를 입력하면 **실행 중인 컨테이너만 보여줍니다.** 그러나 우분투, 파이썬 컨테이너는 실행 중인 컨테이너가 아니기 때문에 목록에서 확인할 수 없는 것입니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container ls -a
CONTAINER ID   IMAGE           COMMAND       CREATED             STATUS                         PORTS     NAMES
4f3c8fa11557   python:3.11.6   "python3"     4 minutes ago       Exited (0) 4 minutes ago                 amazing_shannon
7c9e2ccc46c4   ubuntu          "/bin/bash"   6 minutes ago       Exited (0) 6 minutes ago                 admiring_lamarr
6fffd1f58214   hello-world     "/hello"      About an hour ago   Exited (0) About an hour ago             laughing_maxwell

admin@BGR_AI C:\Users\admin\Desktop>
```

위와 같이 명령어에 **-a** 옵션을 주면 실행 중인 컨테이너와 정지 상태인 컨테이너 모두를 확인할 수 있습니다. 결과를 보면 앞서 `hello-world` , `ubuntu` , `python:3.11.6` 이미지를 실행한 컨테이너를 확인할 수 있습니다. 각 컨테이너는 CONTAINER ID라는 것을 갖는데, 이는 하나의 이미지로 다수의 컨테이너를 생성할 수 있으므로 각 컨테이너는 CONTAINER ID를 갖는 것입니다. 세 개의 컨테이너의 상태(STATUS) 를 보면 Exited (0) 인 것을 알 수 있습니다. 상태가 Exited 라는 것은 **컨테이너가 종료**되었다는 뜻이고 숫자 **0은 정상적으로 종료**되었다는 것을 의미합니다. 그리고 컨테이너가 종료된 이유는 컨테이너를 실행했을 때 컨테이너 내부 프로세스가 모두 종료되면 해당 컨테이너 역시 종료되기 때문입니다.

## Connect inner container

실행 중인 컨테이너 내부에 접속하려면 다음과 같이 컨테이너를 실행할때 **-it** 옵션을 활용하면 됩니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container run -it ubuntu
root@a373a95b4119:/# ls
bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
root@a373a95b4119:/# 
```

- -it 옵션에서 i는 interactive 의 줄임말로 표준 입력(STDIN)을 열어놓는다는 의미이며, t는 tty의 줄임말로 가상터미널을 의미합니다. 즉 -it 옵션을 활용하면 가상 터미널을 통해 키보드 입력을 표준 입력으로 컨테이너에 전달할 수 있는 것입니다.
- 사용자 이름과 호스트 이름이 변경된 것을 알 수 있습니다. 이때 사용자 이름은 root 이고 호스트 이름은 CONTAINER ID 인 것을 알 수 있습니다. ls 명령어를 입력하면 내부의 파일 시스템을 확인할 수 있습니다.

## Exit the container

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container ls
CONTAINER ID   IMAGE     COMMAND       CREATED         STATUS         PORTS     NAMES
a373a95b4119   ubuntu    "/bin/bash"   2 minutes ago   Up 2 minutes             nervous_dijkstra
```

컨테이너 내부에 접속 중인 컨테이너를 종료하는 방법은 크게 두 가지로
1. 컨테이너가 실행중인 터미널에 **exit** 명령어를 통해 컨테이너 밖으로 나가는 방법
2. 다른 터미널로 **docker container stop [컨테이너 ID]** 를 통해 실행 중인 컨테이너를 종료

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container stop a373a95b4119
a373a95b4119

admin@BGR_AI C:\Users\admin\Desktop>
```

여기서 두 번째 방법인 컨테이너 ID를 활용해서 종료하겠습니다. **docker container stop [컨테이너 ID]** 를 입력하면 약 10초 후에 컨테이너가 종료되는 것을 볼 수 있습니다. 참고로 **docker container stop** 과 비슷한 명령어로 **docker container kill** 이 있는데 stop은 약 10초 후에 컨테이너가 종료되고 kill은 즉시 종료됩니다. stop 명령어가 kill 명령어보다 안정성 면에서 효율적이므로 stop 명령어를 사용하길 권장합니다.

## Execute into exited container

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container start a373a95b4119
a373a95b4119

admin@BGR_AI C:\Users\admin\Desktop>docker container ls
CONTAINER ID   IMAGE     COMMAND       CREATED         STATUS         PORTS     NAMES
a373a95b4119   ubuntu    "/bin/bash"   9 minutes ago   Up 6 seconds             nervous_dijkstra

admin@BGR_AI C:\Users\admin\Desktop>docker container attach a373a95b4119
root@a373a95b4119:/# exit
exit

admin@BGR_AI C:\Users\admin\Desktop>
```

종료된 컨테이너를 다시 접속하고 싶다면 다음과 같이 진행합니다.

- **start** 명령어를 이용해 컨테이너를 실행합니다.
- 실행 중인 컨테이너를 확인할 수 있습니다.
- **attach** 명령어를 이용해 내부에 접속할 수 있습니다.
- 접속을 종료하여 실습을 마칩니다.

## Remove the container

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container ls -a
CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS                         PORTS     NAMES
a373a95b4119   ubuntu          "/bin/bash"   46 minutes ago   Exited (0) 5 minutes ago                 nervous_dijkstra
4f3c8fa11557   python:3.11.6   "python3"     55 minutes ago   Exited (0) 22 minutes ago                amazing_shannon
7c9e2ccc46c4   ubuntu          "/bin/bash"   57 minutes ago   Exited (0) 24 minutes ago                admiring_lamarr
6fffd1f58214   hello-world     "/hello"      2 hours ago      Exited (0) About an hour ago             laughing_maxwell
```

전체 컨테이너 목록을 확인합니다. 위와 같은 컨테이너 목록 중 컨테이너 ID가 `6fffd1f58214` 인 컨테이너를 삭제하겠습니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker container rm 6fffd1f58214
6fffd1f58214

admin@BGR_AI C:\Users\admin\Desktop>docker container ls -a
CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS                      PORTS     NAMES
a373a95b4119   ubuntu          "/bin/bash"   47 minutes ago   Exited (0) 38 minutes ago             nervous_dijkstra
4f3c8fa11557   python:3.11.6   "python3"     56 minutes ago   Exited (0) 56 minutes ago             amazing_shannon
7c9e2ccc46c4   ubuntu          "/bin/bash"   58 minutes ago   Exited (0) 58 minutes ago             admiring_lamarr
```

**docker container rm [컨테이너 ID]** 를 입력하면 해당 컨테이너 ID에 해당하는 컨테이너를 삭제할 수 있습니다.

만약 컨테이너 다수를 한 번에 삭제하고 싶다면 컨테이너 ID를 여러 개 입력하면 됩니다.

## Remove the image

이미지를 삭제하려면 **docker image rm [이미지 이름]** 을 입력하면 됩니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker image ls
REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
ubuntu        latest    edbfe74c41f8   3 weeks ago     78.1MB
python        3.11.6    0dba5a08d425   10 months ago   1.01GB
hello-world   latest    d2c94e258dcb   16 months ago   13.3kB
```

이미지 목록을 확인합니다. 위와 같은 이미지 중 hello-world 이미지를 삭제하겠습니다.

```bash
admin@BGR_AI C:\Users\admin\Desktop>docker image rm d2c94e258dcb 
Untagged: hello-world:latest
Untagged: hello-world@sha256:53cc4d415d839c98be39331c948609b659ed725170ad2ca8eb36951288f81b75
Deleted: sha256:d2c94e258dcb3c5ac2798d32e1249e42ef01cba4841c2234249495f87264ac5a
Deleted: sha256:ac28800ec8bb38d5c35b49d45a6ac4777544941199075dff8c4eb63e093aa81e

admin@BGR_AI C:\Users\admin\Desktop>docker image ls
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
ubuntu       latest    edbfe74c41f8   3 weeks ago     78.1MB
python       3.11.6    0dba5a08d425   10 months ago   1.01GB
```

## docker image command list

| 명령어                  | 설명                                     |
| -------------------- | -------------------------------------- |
| docker image build   | [[Dockerfile]] 로부터 이미지를 빌드합니다.         |
| docker image history | 이미지 히스토리를 확인합니다.                       |
| docker image import  | 파일 시스템 이미지 생성을 위한 tarball 콘텐츠를 임포트합니다. |
| docker image inspect | 이미지 정보를 표시합니다.                         |
| docker image load    | tarball로 묶인 이미지를 로드합니다.                |
| docker image ls      | 이미지 목록을 확인합니다.                         |
| docker image prune   | 사용하지 않는 이미지를 삭제합니다.                    |
| docker image pull    | 레지스트리로부터 이미지를 다운로드합니다.                 |
| docker image push    | 레지스트리로 이미지를 업로드합니다.                    |
| docker image rm      | 하나 이상의 이미지를 삭제합니다.                     |
| docker image save    | 이미지를 tarball로 저장합니다.                   |
| docker image tag     | 이미지 태그를 생성합니다.                         |

