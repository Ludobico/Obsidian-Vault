
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

## Git config option

Git을 사용하기 위해서 해야 할 일이 더 있습니다. **git config** 명령을 사용해서 Git 옵션을 설정해야 합니다.

| git config --global <옵션명>             | 지정한 전역 옵션의 내용을 살펴봅니다.    |
| ------------------------------------- | ------------------------ |
| git config --global <옵션명> <새로운 값>     | 지정한 전역 옵션의 값을 새로 설정합니다.  |
| git config --global --unset <옵션명>     | 지정한 전역 옵션을 삭제합니다.        |
| git config --local <옵션명>              | 지정한 로컬 옵션의 내용을 살펴봅니다.    |
| git config --local <옵션명> <새로운 값>      | 지정한 로컬 옵션의 값을 새로 설정합니다.  |
| git config --local --unset <옵션명>      | 지정한 로컬 옵션을 삭제합니다.        |
| git config --system <옵션명>             | 지정한 시스템 옵션의 내용을 살펴봅니다.   |
| git config --system <옵션명> <새로운 값>     | 지정한 시스템 옵션의 값을 새로 설정합니다. |
| git config --system --unset <옵션명> <값> | 지정한 시스템 옵션의 값을 삭제합니다.    |
| git config --list                     | 현재 프로젝트의 모든 옵션을 살펴봅니다.   |

**git config** 명령으로는 옵션을 보거나, 값을 바꿀 수 있습니다. Git 옵션에는 <font color="#ffff00">지역(local) 옵션</font>과 전<font color="#ffff00">역(global) 옵션</font>, <font color="#ffff00">시스템(system) 환경 옵션</font>의 세 종류가 있습니다. <font color="#ffff00">시스템 환경 옵션은 PC 전체의 사용자를 위한 옵션</font>, <font color="#ffff00">전역 옵션은 현재 사용자를 위한 옵션</font>이고, <font color="#ffff00">지역 옵션은 Git 저장소에서만 유효한 옵션</font>입니다. 우선순위는 지역 옵션 > 전역 옵션 > 시스템 옵션 순으로 지역 옵션이 가장 높습니다. 일반적으로 개인 PC에서는 전역 옵션을 많이 사용하는데, 공용 PC처럼 여러 사람이 사용하거나 Git을 잠깐만 써야 할 일이 있다면 지역 옵션을 사용해야 합니다. 시스템 옵션은 Git 설치 시에 몇 가지 값들이 지정되는데 직접 수정하는 일은 그리 많지 않습니다.

옵션값을 이용하면 여러 가지 설정이 가능합니다. 지금은 필수 설정인 **user.name** , **user.email** , **core.editor** 세 옵션의 값을 입력해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config --global user.name
Ludobico

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config --global user.name "MA SADIK"

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config --global user.name
MA SADIK
```

> git config --global user.name 명령을 실행했을 때 아무런 결과도 출력되지 않는다면 현재 설정된 값이 없다는 의미입니다.

중요한 설정이 하나 더 남아 있습니다. CLI를 사용하면 텍스트 에디터를 쓸 일이 생기는데, 현재 Git Bash의 기본 에디터는 보통 리눅스 운영체제에서 주로 쓰이는 vim이나 nano로 설정되어 있습니다. 기본 에디터를 VS 코드로 변경하는 것이 좋습니다. 만약 이미 VS 코드가 기본 에디터로 되어 있다면 그대로 두면 됩니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config core.editor

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config --global core.editor

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config --system core.editor
```

마지막으로 **user.name** 값과 **color.ui** 값도 설정해 보겠습니다. user.email 값에는 GitHub를 가입할 때 사용한 이메일을 지정하고, color.ui 값에는 가독성을 위해 auto로 지정합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config user.email "aqs450@gmail.com"

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git config color.ui auto
```

## add & commit

먼저 기본 Git 명령어를 살펴보겠습니다.

| git add <파일명> ...                        | 파일들을 스테이지에 추가합니다. 새로 생성한 파일을 스테이지에 추가하고 싶다면 반드시 add 명령어를 사용합니다.                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| git commit                               | 스테이지에 있는 파일들을 커밋합니다.                                                                                   |
| git commit -a                            | add 명령어를 생략하고 바로 커밋하고 싶을 때 사용합니다. 변경된 파일과 삭제된 파일은 자동으로 스테이징되고 커밋됩니다. 주의할 점은 untracked 파일은 커밋되지 않습니다.   |
| git push \[-u\] \[원격 저장소 별명\] \[브랜치 이름\] | 현재 브랜치에서 새로 생성한 커밋들을 원격 저장소에 업로드합니다. -u 옵션으로 브랜치의 업스트림을 등록할 수 있습니다. 한 번 등록한 후에는 git push 명령만 입력해도 됩니다. |
| git pull                                 | 원격 저장소의 변경 사항을 워킹트리에 반영합니다. 사실은 git fetch + git merge 명령입니다.                                           |
| git fetch \[원격 저장소 별명\] \[브랜치 이름\]       | 원격 저장소의 브랜치와 커밋들을 로컬 저장소와 동기화합니다. 옵션을 생략하면 모든 원격 저장소에서 모든 브랜치를 가져옵니다.                                  |
| git merge <대상 브랜치>                       | 지정한 브랜치의 커밋들을 현재 브랜치 및 워킹트리에 반영합니다.                                                                    |

커밋을 실행하기 위해 먼저 간단한 파일을 만들겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ echo "hello git"
hello git

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ echo "hello git" > file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ ls
file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git status
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        file1.txt

nothing added to commit but untracked files present (use "git add" to track)
```

**git status** 명령으로 상태를 살펴보니 file1.txt 이라는 파일이 생성되었고 untracked 상태임을 확인할 수 있습니다. 또한 git add \<file\> ... 명령을 사용하면 커밋에 포함될 수 있다는 내용도 볼 수 있습니다.

> ... 은 한 번에 여러 파일 이름을 지정할 수도 있다는 뜻입니다.

이번에는 변경 내용을 **git add** 명령으로 스테이지에 추가해 보겠습니다. 파일을 스테이지에 올린다하여 <font color="#ffff00">스테이징</font>이라고도 합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git add file1.txt 
warning: in the working copy of 'file1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   file1.txt
```

file1.txt 파일이 스테이지 영역에 추가된 것을 확인할 수 있습니다.

### reset : unstage

위 실행 결과의 마지막에서 두 번째 줄을 보면 아래와 같은 커맨드가 있습니다.

```bash
(use "git rm --cached <file>..." to unstage)
```

이 명령으로 스테이지에서 내릴수 있다(unstage)는 메시지가 있습니다. 그런데 스테이지에서 내리기 위해 저 명령보다 자주 사용하는 명령이 있습니다. 바로 **git reset** 명령인데요, 이것을 사용하면 더 쉽게 파일을 스테이지에서 내릴 수 있습니다.

| git reset \[파일명\] ... | 스테이지 영역에 있는 파일을 스테이지에서 내립니다(unstaging). 워킹트리의 내용은 변경되지 않습니다. 옵션을 생략할 경우 스테이지의 모든 변경 사항을 초기화합니다. |
| --------------------- | ----------------------------------------------------------------------------------------------- |

이 명령은 워킹트리의 내용은 그대로 두고 해당 파일을 스테이지에서만 내립니다. 세 가지 옵션 (<font color="#ffff00">soft, mixed, hard</font>) 을 사용할 수 있는데 지금처럼 옵션 없이 사용하면 mixed reset 으로 동작합니다. 이렇게 스테이지에서 내리는 작업을 언스테이징(unstaging) 이라고 합니다.

다음은 `file1.txt` 를 **git reset** 명령으로 언스테이징하고, cat 명령어로 파일 내용이 변경되었는지 확인합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   file1.txt


Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git reset file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git status
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        file1.txt

nothing added to commit but untracked files present (use "git add" to track)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ ls
file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ cat file1.txt 
hello git
```

파일의 내용은 그대로 두고 단지 언스테이징만 진행한 것을 알 수 있습니다.

### create commit

이제 커밋을 실행해 보겠습니다. 좀 전에 언스테이징을 했으므로 다시 **git add** 명령을 실해한 후 커밋합니다. 커밋은 **git commit** 명령으로 수행합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git add file1.txt 
warning: in the working copy of 'file1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git status 
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   file1.txt


Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git commit

```

**git commit** 명령을 실행하면 다음 그림과 같이 git bash의 기본 코어에디터인 [[vim]]이 열립니다

```vim
# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# On branch main
#
# Initial commit
#
# Changes to be committed:
#       new file:   file1.txt
#
.git/COMMIT_EDITMSG [unix] (10:36 05/09/2024)
```

[[vim]] 명령어를 활용하여 다음 처럼 적어줍니다. 이때 <font color="#ffff00">첫째 줄과 둘째 줄 사이는 반드시 한 줄 비어야 합니다.</font> 그리고 첫 줄에는 작업 요약, 다음 줄에는 작업 내용을 자세하게 기록합니다. 첫 줄은 제목이고 그 다음 줄은 본문이라고 생각하면 됩니다. 로그를 볼 때나 Pull request 메뉴 등에서 이 규치을 활용해서 내용을 자동으로 구성하기 때문에 꼭 지키는 것이 좋습니다.

```vim
첫 번째 커밋

간단하게 hello git이라고 쓴 내용을 커밋했다.

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# On branch main
#
# Initial commit
#
# Changes to be committed:
#       new file:   file1.txt
```

```bash
[main (root-commit) 8217549] 첫 번째 커밋
 1 file changed, 1 insertion(+)
 create mode 100644 file1.txt
```

만약 **git commit** 명령을 실행한 후 갑작스런 변심 등의 이유로 커밋을 하고 싶지 않다면 vim을 기준으로 :q! 명령으로 종료합니다. 그럼 커밋도 자동으로 취소됩니다.

커밋을 성공했다면 **git status** 명령을 실행해 상태를 확인하면 스테이지 영역이 깨끗해진 걸 확인할 수 있을 것입니다.

```bash
$ git status
On branch main
nothing to commit, working tree clean
```

커밋이 만들어지면 그 커밋 시점의 파일 상태로 언제라도 복구할 수 있습니다. 그리고 커밋은 절대 사라지지 않습니다.

> 좋은 커밋 메시지의 일곱 가지 규칙
> 1. 제목과 본문을 빈 줄으로 분리합니다.
> 2. 제목은 50자 이내로 씁니다.
> 3. 제목을 영어로 쓸 경우 첫 글자는 대문자로 씁니다.
> 4. 제목에는 마침표를 넣지 않습니다.
> 5. 제목을 영어로 쓸 경우 동사원형(현재형)으로 시작합니다.
> 6. 본문을 72자 단위로 줄바꿈합니다.
> 7. 어떻게 보다 무엇과 왜를 설명합니다.

### log : check the commit history

**git log** 명령으로 Git의 커밋 히스토리를 확인해 보겠습니다.

| git log                                    | 현재 브랜치의 커밋 이력을 보는 명령입니다.                                                                                                                                                                                                                                                    |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| git log -n <숫자>                            | 전체 커밋 중에서 최신 n개의 커밋만 살펴봅니다. 아래의 다양한 옵션과 조합해서 쓸 수 있습니다.                                                                                                                                                                                                                      |
| git log --oneline --graph --all --decorate | 자주 사용하는 옵션입니다. 로그를 간결하게 보여줍니다.<br><br>--oneline : 커밋 메시지를 한 줄로 요약해서 보여줍니다. 생략하면 커밋 정보를 자세히 표시합니다.<br><br>--graph : 커밋 옆에 브랜치의 흐름을 그래프로 보여 줍니다. GUI와 유사한 모습으로 나옵니다.<br><br>--all : all 옵션을 지정하지 않으면 HEAD와 관계없는 옵션은 보여 주지 않습니다.<br><br>--decorate : 브랜치와 태그 등의 참조를 간결히 표시합니다. |

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --graph --all --decorate
* 8217549 (HEAD -> main) 첫 번째 커밋
```

커밋 히스토리에 보이는 앞의 16진수 7자리 숫자는 <font color="#ffff00">커밋 체크섬</font> 또는 <font color="#ffff00">커밋 아이디</font> 입니다. SHA1 해시 체크섬 값을 사용하는데, 전 세계에서 유일한 값을 가집니다. 실제로 커밋 체크섬은 40자리인데 앞의 7자리만 화면에 보여 줍니다.

## help

Git에는 각 명령의 도움말을 볼 수 있는 명령이 있습니다. 모르는 명령이 있거나 그 명령의 자세한 옵션들이 보고 싶을 때에는 **git help** 명령을 사용하면 됩니다.

| git help <명령어> | 해당 명령어의 도움말을 표시합니다. |
| -------------- | ------------------- |
**help** 명령을 수행하면 웹 브라우저가 열리면서 다음과 같이 해당 명령어에 대한 내용이 표시됩니다.

![[Pasted image 20240905142010.png]]

## remote and push

