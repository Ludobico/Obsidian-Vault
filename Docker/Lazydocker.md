![[Pasted image 20250402110053.png]]



[Lazydocker](https://github.com/jesseduffield/lazydocker) 는 [[Docker]] 및 [[Docker compose]] 를 **터미널 환겨에서 쉽고 직관적으로 관리할 수 있도록 도와주는 도구**입니다. Docker 명령어를 외울필요없이, 시각적인 인터페이스를 통해 컨테이너, 이미지, 볼륨, 네트워크 등을 제어할 수 있습니다.

## Installation
---

### Windows

- Chocolatey가 설치되어 있어야합니다. 다음 요구사항을 확인하세요.
	- Windows 7+
	- PowerShell v2+
	- .NET Framerwork 4+
- 설치되지 않았다면 PowerShell에서 아래의 명령어로 설치 및 확인을 진행합니다.

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

```powershell
choco
```

```
Chocolatey v2.2.2
Please run 'choco -?' or 'choco <command> -?' for help menu.
```


```shell
choco install lazydocker
```

