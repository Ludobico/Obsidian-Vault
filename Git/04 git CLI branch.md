
## Relationshipt between commit and branch

1. 커밋하면 커밋 객체가 생깁니다. 커밋 객체에는 부모 커밋에 대한 참조와 실제 커밋을 구성하는 파일 객체가 들어 있습니다.
2. 브랜치는 논리적으로는 어떤 커밋과 그 조상들을 묶어서 뜻하지만, 사실은 단순히 커밋 객체 하나를 가리킬 뿐입니다.

### branch : build the branch

브랜치를 다루는 명령과 옵션을 살펴보면 다음과 같습니다.

| git branch \[-v\]                   | 로컬 저장소의 브랜치 목록을 보는 명령으로 -v 옵션을 사용하면 마지막 커밋도 함께 표시됩니다.<br>표시된 브랜치 중 이름 왼쪽에 \*가 붙어있으면 HEAD 브랜치입니다.    |
| ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| git branch \[-f\] <브랜치 이름> <커밋 체크섬> | 새로운 브랜치를 생성합니다. 커밋 체크섬 값을 주지 않으면 HEAD 로부터 브랜치를 생성합니다. 이미 있는 브랜치를 다른 커밋으로 옮기고 싶을 때는 -f 옵션을 사용해야 합니다. |
| git branch -r\[v]                   | 원격 저장소에 있는 브랜치를 보고 싶을 때 사용합니다. 마찬가지로 -v 옵션을 추가하여 커밋 요약도 볼 수 있습니다.                                   |
| git switch <브랜치 이름>                 | 특정 브랜치로 변경할 때 사용합니다.                                                                                |
| git switch -c <브랜치 이름> \[커밋 체크섬\]   | 특정 커밋에서 브랜치를 새로 생성하고 동시에 변경까지 합니다. 두 명령을 하나로 합친 명령이기 때문에 간결해서 자주 사용합니다.                             |
| git merge <대상 브랜치>                  | 현재 브랜치와 대상 브랜치를 병합할 때 사용합니다. 병합 커밋이 새로 생기는 경우가 많습니다.                                                |
| git rebase <대상 브랜치>                 | 현재 브랜치의 커밋을 대상 브랜치에 재배치합니다. 히스토리가 깔끔해져서 자주 사용하지만 조심해서 사용해야 합니다.                                     |
| git branch -d <브랜치 이름>              | 특정 브랜치를 삭제할 때 사용합니다. HEAD 브랜치나 병하빙 되지 않은 브랜치는 삭제할 수 없습니다.                                           |
| git branch -D <브랜치 이름>              | 브랜치를 강제로 삭제하는 명령입니다. -d로 삭제할 수 없는 브랜치를 지우고 싶을 때 사용합니다.                                              |

이번에는 새로운 브랜치를 만들고 두 번 커밋한 후에 다시 \[main\] 브랜치로 병합해 보겠습니다.

현재 브랜치를 확인하고 새로운 브랜치를 만들 수 있는 **git branch** 명령을 실행해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline
991cb7e (HEAD -> main, origin/main) 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git branch
* main

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git branch mybranch1

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git branch
* main
  mybranch1

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --all
991cb7e (HEAD -> main, origin/main, mybranch1) 두 번째 커밋
8217549 첫 번째 커밋
```

> 1. HEAD는 현재 작업 중인 블ㄴ치를 가리킵니다.
> 2. 브랜치는 커밋을 가리키므로 HEAD도 커밋을 가리킵니다.
> 3. 결국 HEAD는 현재 작업 중인 브랜치의 최근 커밋을 가리킵니다.

순서대로 알아보면

- **git log** 명령을 통해 현재 커밋과 브랜치의 상태를 확인합니다. \[origin\]으로 시작하는 브랜치는 원격 브랜치이므로 현재 로컬에는 \[main\] 브랜치만 존재하는 것을 알 수 있습니다. 그리고 HEAD가 \[main\] 브랜치를 가리키는 것도 확인할 수 있습니다.
- **git branch** 명령을 수행 시 \*main 문구는 HEAD -> main 과 동일한 의미입니다. 그리고 HEAD가 \[main\] 브랜치를 가리키는 것도 확인할 수 있습니다.
- **git branch mybranch1** 명령을 통해 새로운 브랜치인 \[mybranch1\] 브랜치를 생성합니다.
- **git branch** 와 **git log** 명령으로 결과를 확인합니다. 가장 최신 커밋인 `991cb7e` 커밋에 HEAD, main, mybranch1 모두 위치하고 있는 것을 알 수 있습니다. 아직 체크아웃 전이기 때문에 여전히 HEAD는 \[main\] 브랜치르를 가리키고 있습니다.

### switch : change the branch

CLI 환경에서는 브랜치를 이동할 때 **git checkout** 명령 대신 **git switch** 명령을 사용하겠습니다. \[main\] 브랜치에서 \[mybranch1\] 브랜치로 변경하고, 새로운 커밋을 생성한 뒤에 결과를 확인해 봅니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git switch mybranch1
Switched to branch 'mybranch1'

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git branch
  main
* mybranch1

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git log --oneline --all
991cb7e (HEAD -> mybranch1, origin/main, main) 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ cat file1.txt 
hello git
second

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ echo "third - my branch" >> file1.txt 

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ cat file1.txt 
hello git
second
third - my branch

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git status
On branch mybranch1
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   file1.txt

no changes added to commit (use "git add" and/or "git commit -a")

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git add file1.txt 
warning: in the working copy of 'file1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git commit
[mybranch1 d424559] mybranch1 첫 번째 커밋
 1 file changed, 1 insertion(+)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git log --oneline --all --graph --decorate
* d424559 (HEAD -> mybranch1) mybranch1 첫 번째 커밋
* 991cb7e (origin/main, main) 두 번째 커밋
* 8217549 첫 번째 커밋
```

위 실행 과정을 설명하면 다음과 같습니다.

1. **git switch** 명령을 수행해서 브랜치를 변경합니다. 보통 브랜치 변경과 동시에 작업 폴더의 내용도 변경되는데, 이번에는 `mybranch1` 커밋이 이전 브랜치였던 \[main\]의 커밋과 같은 커밋이라서 작업 폴더의 내용은 변경되지 않습니다.
2. **git branch** 명령으로 현재 브랜치를 확인합니다.
3. **git log** 명령을 통해 HEAD가 \[mybranch1\] 으로 변경된 것을 확인할 수 있습니다. 여기서 자세보면 **git branch** 명령에서도 HEAD가 `mybranch1` 으로 변경된 것을 확인할 수 있습니다.
4. 커밋 메시지를 "mybranch1 첫 번째 커밋" 이라고 입력해 새로운 커밋을 생성합니다.
5. 커밋 히스토리를 확인합니다.

> 1. 새로 커밋을 생성하면 그 커밋의 부모는 언제나 이전 HEAD 커밋입니다.
> 2. 커밋이 생성되면 HEAD는 새로운 커밋으로 갱신됩니다.
> 3. HEAD가 가리키는 브랜치도 HEAD와 함께 새로운 커밋을 가리킵니다.


커밋 전과 커밋 후의 상태는 다음 그림과 같습니다.

![[Pasted image 20240906093626.png]]
<커밋 전>

![[Pasted image 20240906093730.png]]
<커밋 후>
### fast-forward merge

이번에는 CLI에서 병합해 보겠습니다. 브랜치를 병합하는 명령은 **git merge** 입니다. 별도로 옵션을지정하지 않으면 기본적으로 빨리 감기 병합(fast forward merge)이 수행됩니다.

\[mybranch1\] 브랜치에서 파일을 수정한 후 추가로 한 번 더 커밋하겠습니다. 그다음 \[main\] 브랜치로 변경해 \[main\] 브랜치와 \[mybranch1\] 브랜치를 병합해 보겠습니다.

![[Pasted image 20240906093828.png]]
<커밋 전>

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ echo "fouth - my branch" >> file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ cat file1.txt
hello git
second
third - my branch
fouth - my branch

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git status
On branch mybranch1
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   file1.txt

no changes added to commit (use "git add" and/or "git commit -a")

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git add file1.txt
warning: in the working copy of 'file1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git commit
[mybranch1 5761085] mybranch1 두 번째 커밋
 1 file changed, 1 insertion(+)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git log --oneline --all --graph
* 5761085 (HEAD -> mybranch1) mybranch1 두 번째 커밋
* d424559 mybranch1 첫 번째 커밋
* 991cb7e (origin/main, main) 두 번째 커밋
* 8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git switch main
Switched to branch 'main'
Your branch is up to date with 'origin/main'.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ cat file1.txt 
hello git
second

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git merge mybranch1
Updating 991cb7e..5761085
Fast-forward
 file1.txt | 2 ++
 1 file changed, 2 insertions(+)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --all --graph
* 5761085 (HEAD -> main, mybranch1) mybranch1 두 번째 커밋
* d424559 mybranch1 첫 번째 커밋
* 991cb7e (origin/main) 두 번째 커밋
* 8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ cat file1.txt 
hello git
second
third - my branch
fouth - my branch
```

![[Pasted image 20240906102151.png]]
<커밋 후>

위 명령에 대해 살펴보면 다음과 같습니다.

1. 커밋 메시지를 "mybranch1 두 번째 커밋"으로 입력해 커밋합니다. 그 결과 새로운 커밋인 `5761085` 를 생성했습니다.
2. 로그를 보면 기존 커밋을 부모로 하는 새로운 커밋이 생성되었습니다. 그리고 HEAD는 `mybranch1`  mybranch1은 새 커밋을 각각 가리키는 것을 확인할 수 있습니다.
3. **git switch** 명령을 이용해서 \[main\] 브랜치로 변경합니다.
4. cat 명령어를 통해 텍스트 파일의 내용이 이전으로 돌아간 것을 확인할 수 있습니다.
5. \[main\] 브랜치에 \[mybranch1\] 브랜치를 병합합니다. 여기에 **git merge** 명령이 사용됩니다.

이번 병합은 작업의 흐름이 하나였기 때문에 예상했던 것처럼 빨리 감기 병합으로 완료되었습니다.

