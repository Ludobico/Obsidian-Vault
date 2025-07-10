
## 도커 서비스 상태 확인

```
Get-Service -Name "com.docker.service"
```

```
Status   Name               DisplayName
------   ----               -----------
Stopped  com.docker.service Docker Desktop Service
```

## 도커 서비스 시작

```
net start com.docker.service
```

