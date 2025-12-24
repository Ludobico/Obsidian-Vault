GitHub 계정이 2FA가 활성화 되어 있으면 아이디,패스워드로 접근이 되지 않습니다.

리눅스에서 [[Git]] 을 사용할 때는 인증을 해야 하는데, 아이디는 깃허브의 User.name을 입력하고 비밀번호는 **token 을 사용하여 인증** 해야 합니다.

## git clone 명령어

```git
git clone https://<user-id>:<ghp_token>@<repository url>
```

## History management

이렇게 clone 받을 경우 토큰정보가  **Shell history에 남기 때문에 삭제를 권장** 합니다. 

```bash
history
```

```bash
 1994  git config user.email
 1995  sudo apt update
 1996  git clone https://github.com/develop-aihops/chosun-Univ-RAG.git
 1997  git config --global credential.helper store
 1998  git clone https://github.com/develop-aihops/chosun-Univ-RAG.git
 1999  git clone https://Ludobico:ghp_<token>@github.com/develop-aihops/repository.git
 2000  history
 ...
```

```bash
history -d 1999
```

```bash
1997  git config --global credential.helper store
 1998  git clone https://github.com/develop-aihops/chosun-Univ-RAG.git
 1999  history
 2000  history -d 1999
 2001  history
 ...
```

혹은 `grep` 명령어로 토큰의 앞글자를 찾아서 삭제하는 방법도 가능합니다.

```bash
history | grep ghp_
```

`history -d` 는 메모리에서만 지워집니다. 파일에도 반영하려면

```bash
history -w
```

명령어를 사용하거나 세션 종료시 자동 저장됩니다.

