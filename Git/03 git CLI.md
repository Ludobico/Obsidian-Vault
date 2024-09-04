
## git bash basic

기본적으로 `Git Bash` 를 시작하면 현재 폴더는 사용자의 홈 폴더에서 시작합니다. 홈 폴더의 전체 경로는 윈도우 기준으로 `c:/Users/사용자ID` 가 되는데 이를 줄여서 `~` 로 나타내는 것입니다.

```
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (main)
$ 
```

> 프롬프트 끝에 브랜치명이 보인다면 이는 Git 작업 폴더라는 의미입니다.

처음 실행할때 나타내는 프롬프트는 현재 사용자의 정보와 Git bash의 환경을 보여줍니다.  위 프롬프트의 의미는 다음과 같습니다.

<font color="#ffff00">Ludobico@Ludobico</font>
- Ludobico(첫 번째 부분) : 현재 로그인한 사용자 이름입니다. 
- Ludobico(두 번째 부분) : 현재 사용하는 컴퓨터의 호스트 이름입니다. 이는 사용자가 설정한 컴퓨터이름입니다.

<font color="#ffff00">MINGW64</font>
- Minimalist GNU for Windows 64-bit의 약자로, <font color="#ffff00">현재 Git Bash가 사용하고 있는 쉘 환경</font>을 나타냅니다.

| pwd              | 현재 폴더의 위치를 확인합니다.                          |
| ---------------- | ------------------------------------------ |
| ls -a            | 현재 폴더의 파일 목록을 확인하며 -a 인자로 숨김 파일도 볼 수 있습니다. |
| cd               | 홈 폴더로 이동합니다.                               |
| cd <폴더 이름>       | 특정 위치의 디렉토리로 이동합니다.                        |
| cd ../           | 현재 폴더의 상위 폴더로 이동합니다.                       |
| mkdir <새 폴더 이름>  | 현재 폴더의 아래에 새로운 폴더를 만듭니다.                   |
| echo "Hello Git" | 화면에 "" 안의 문장을 표시합니다.                       |

## Create local repository and check the status

[[Git]] 로컬 저장소를 만들기 위해 mkdir 명령어로 \[hello-git-cli\] 라는 폴더를 만들고 이동합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub
$ mkdir hello-git-cli

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub
$ cd hello-git-cli/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli
$ pwd
/c/Users/aqs45/OneDrive/Desktop/repoSub/hello-git-cli
```

이제 **git status** 명령으로 새로 만든 폴더 정보를 확인해 보겠습니다. git status는 Git 저장소의 상태를 확인할 수 있기 때문에 자주 사용됩니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli
$ git status
fatal: not a git repository (or any of the parent directories): .git
```

그런데 막상 git status 명령을 실행하면 오류가 발생합니다. 오류 메시지를 보면 `.git 폴더가 없다` 라고 알려줍니다. 즉, git status 명령은 Git 작업 폴더(working tree) 에서만 정상적으로 수행되는 명령입니다. 

| git status    | Git working tree의 상태를 보는 명령어로 매우 자주 사용합니다. 워킹트리가 아닌 폴더에서 실행하면 오류가 발생합니다. |
| ------------- | ------------------------------------------------------------------------ |
| git status -s | git status 명령보다 짧게 요약해서 상태를 보여 주는 명령으로 변경된 파일이 많을 때 유용합니다.               |

이제 우리가 만든 폴더를 Git 저장소로 만들어 보겠습니다. 앞서 생성한 \[hello-git-cli\] 폴더에서 다음처럼 실행합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli
$ git init -b main
Initialized empty Git repository in C:/Users/aqs45/OneDrive/Desktop/repoSub/hello-git-cli/.git/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ ls -a
./  ../  .git/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git status
On branch main

No commits yet

nothing to commit (create/copy files and use "git add" to track)
```

**git init** 명령은 현재 폴더에 Git 저장소를 생성합니다. **-b** 옵션으로 기본 브랜치를 main으로 지정했습니다. 명령의 결과는 비어 있는 Git 저장소를 .git 폴더에 만들어라 라는 내용입니다.

| working tree      | 워크트리, 워킹 디렉터리, 작업 디렉터리, 작업 폴더 모두 같은 뜻으로 사용됩니다. 일반적으로 사용자가 파일과 하위 폴더를 만들고 작업 결과물을 저장하는 곳을 Git에서는 working tree라고 부릅니다. 공식 문서에서는 워킹트리를 커밋을 체크아웃하면 생성되는 파일과 디렉터리로 정의하고 있습니다. 정확하게는 작업 폴더에서 .git 폴더를 뺀 나머지 부분이 워킹트리입니다. |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| local repository  | **git init** 명령으로 생성되는 .git 폴더가 로컬 저장소입니다. 커밋, 커밋을 구성하는 객체, 스테이지가 모두 이 폴더에 저장됩니다.                                                                                                                                  |
| remote repository | 로컬 저장소를 업로드하는 곳을 원격 저장소라고 부릅니다. 우리가 사용하고 있는 GitHub가 원격 저장소입니다.                                                                                                                                                     |
| Git repository    | Git 명령으로 관리할 수 있는 폴더 전체를 일반적으로 Git 프로젝트 혹은 Git 저장소라고 부릅니다. 공식문서에서는 로컬 저장소와 Git 저장소를 같은 뜻으로 사용합니다.                                                                                                                  |

## Git command option

Git을 사용하기 위해서 해야 할 일이 더 있습니다. **git config** 명령을 사용해서 Git 옵션을 설정해야 합니다.

