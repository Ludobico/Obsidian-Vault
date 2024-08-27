
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

## execute docker container

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

