
# Apply Timezone in running container
## caution

아래 방법은 현재 실행 중인 컨테이너에 대해서만 적용됩니다. <font color="#ffff00">컨테이너를 재시작하거나 새로 생성할 경우, 다시 설정</font>해야 합니다. 지속적으로 적용하려면 [[Dockerfile]] 또는 컨테이너 실행 명령에 시간을 설정하는 방법을 사용하는 것이 좋습니다.


1. 먼저 실행 중인 컨테이너에 접속합니다. `CONTAINER_ID` 또는 `CONTAINER_NAME` 을 사용하여 접근 할 수 있습니다.

```bash
docker exec -it <CONTAINER_ID> /bin/bash
```

2. 컨테이너에서 시간대를 변경하려면 `tzdata` 패키지가 필요할 수 있습니다. 패키지를 설치합니다.

```bash
apt-get update && apt-get install -y tzdata
```

3. 아래 명령어를 사용하여 시간대를 `Asia/Seoul` 로 설정합니다.

```bash
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
echo "Asia/Seoul" > /etc/timezone
```

4. 변경된 시간대를 확인하여 제대로 적용되었는지 확인합니다.

```bash
date
```

