
## 도커 서비스 상태 확인

```
Get-Service -Name "com.docker.service"
```

```
Status   Name               DisplayName
------   ----               -----------
Stopped  com.docker.service Docker Desktop Service
```

## wsl 재시작

```
wsl -d docker-desktop
```

### wsl 상태 확인

```
wsl --list --verbose
```

```
* Ubuntu            Stopped         2
  docker-desktop    Running         2
```
## 도커 서비스 재시작

```
net stop com.docker.service
net start com.docker.service
```

## 도커 데스크탑 시작

```
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

