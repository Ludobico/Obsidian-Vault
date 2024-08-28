[[Docker]] 에서는 컨테이너의 데이터 저장을 관리하기 위해 여러 가지 스토리지 옵션을 제공합니다. 이 중 주요한 세 가지는 다음과 같습니다.

## 1. Volume

Docker 에서 가장 추천되는 스토리지 옵션입니다. 볼륨은 Docker가 관리하는 호스트 파일 시스템의 특정 위치에 데이터를 저장합니다.

## 2. Bind Mount

호스트 파일 시스템의 특정 디렉토리를 컨테이너에 연결하는 방식입니다. 컨테이너 내부에서 이 디렉토리를 마치 로컬 디렉토리처럼 사용할 수 있습니다.

## 3. Tmpfs Mount

컨테이너의 메모리에서만 데이터를 저장하는 방식입니다. 주로 데이터를 저장할 때 사용됩니다.

---

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

