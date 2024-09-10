- [[#Relationshipt between commit and branch|Relationshipt between commit and branch]]
	- [[#Relationshipt between commit and branch#branch : build the branch|branch : build the branch]]
	- [[#Relationshipt between commit and branch#switch : change the branch|switch : change the branch]]
	- [[#Relationshipt between commit and branch#fast-forward merge|fast-forward merge]]
	- [[#Relationshipt between commit and branch#reset --hrad : reset the branch|reset --hrad : reset the branch]]
	- [[#Relationshipt between commit and branch#rebase|rebase]]
	- [[#Relationshipt between commit and branch#tag|tag]]

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

### reset --hrad : reset the branch

**git reset** 명령은 <font color="#ffff00">현재 브랜치를 특정 커밋으로 되돌릴 때 사용</font>합니다. 이 중에서 많이 사용하는 **git reset --hard** 명령을 실행하면 현재 브랜치를 지정한 커밋으로 옮긴 후 해당 커밋의 내용을 작업 폴더에도 반영합니다.

| git reset --hard <이동할 커밋 체크섬> | 현재 브랜치를 지정한 커밋으로 옮깁니다. 작업 폴더의 내용도 함께 변경됩니다. |
| ----------------------------- | ------------------------------------------- |

**git reset --hard** 명령을 사용하려면 커밋 체크섬을 알아야 합니다. 커밋 체크섬은 **git log**를 통해 확인할 수 있지만 CLI에서 복잡한 커밋 체크섬을 타이핑하는 건 꽤 번거로운 작업입니다. 이럴때 보통 **HEAD~** 또는 **HEAD^** 로 시작하는 약칭을 사용할 수 있습니다.

| HEAD~<숫자> | HEAD~ 은 헤드의 부모 커밋, HEAD~2는 헤드의 Grandparent 커밋을 말합니다. HEAD~n은 n번째 위쪽 조상이라는 뜻입니다.   |
| --------- | --------------------------------------------------------------------------------- |
| HEAD^<숫자> | HEAD^은 똑같이 부모 커밋입니다. 반면 HEAD^2는 두 번째 부모를 가리킵니다. 병합 커밋처럼 부모가 둘 이상인 커밋에서만 의미가 있습니다. |

이번에는 \[main\] 브랜치 커밋을 두 커밋 이전으로 옮겨 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git reset --hard HEAD~2
HEAD is now at 991cb7e 두 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --all
5761085 (mybranch1) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
991cb7e (HEAD -> main, origin/main) 두 번째 커밋
8217549 첫 번째 커밋
```

1. **git reset --hard HEAD~2** 를 실행해서 HEAD를 2단계 이전으로 되돌립니다.
2. **log** 명령으로 확인해 보면 HEAD -> main이 달라진 것을 알 수 있습니다.

이번에 수행한 명령을 그림으로 나타내면 다음과 같습니다.

![[Pasted image 20240906110913.png|256]]
![[Pasted image 20240906111018.png|256]]


### rebase

이번에는 빨리 감기 병합이 가능한 상황에서 **git rebase** 명령을 사용해 보겠습니다. **git rebase <대상 브랜치>** 명령은 <font color="#ffff00">현재 브랜치에만 있는 새로운 커밋을 대상 브랜치 위로 재배치</font>합니다.  그런데 현재 브랜치에 재배치할 커밋이 없을 경우 git rebase 명령은 아무런 동작을 하지 않습니다. 또한 빨리 감기 병합이 가능한 경우에는 git merge 명령처럼 동작합니다.

이를 그림으로 표현하면 다음과 같습니다.

![[Pasted image 20240906134954.png]]
<rebase 이전>

![[Pasted image 20240906135024.png]]
<rebase 이후>

그럼 **git merge** 명령 대신 **git rebase** 명령으로 빨리 감기 병합을 하고 \[mybranch1\] 브랜치를 제거해보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git branch
* main
  mybranch1

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git switch mybranch1
Switched to branch 'mybranch1'

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git rebase main
Current branch mybranch1 up to date.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git log --oneline --all
5761085 (HEAD -> mybranch1) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
991cb7e (origin/main, main) 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (mybranch1)
$ git switch main
Switched to branch 'main'
Your branch is up to date with 'origin/main'.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git rebase mybranch1
Successfully rebased and updated refs/heads/main.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --all
5761085 (HEAD -> main, mybranch1) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
991cb7e (origin/main) 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git push
Enumerating objects: 8, done.
Counting objects: 100% (8/8), done.
Delta compression using up to 16 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (6/6), 586 bytes | 586.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Ludobico/hello-git-cli.git
   991cb7e..5761085  main -> main

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git branch -d mybranch1
Deleted branch mybranch1 (was 5761085).

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --all -n2
5761085 (HEAD -> main, origin/main) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
```

1. \[mybranch1\] 브랜치로 변경합니다.
2. \[mybranch1\] 브랜치는 이미 \[main\] 브랜치 위에 있기 때문에 재배치할 커밋이 없습니다. 그서 **git rebase main** 을 수행해도 아무 일도 일어나지 않습니다.
3. 다시 \[main\] 브랜치로 변경합니다.
4. **git rebase** 명령으로 \[main\] 브랜치를 \[mybranch1\] 브랜치로 재배치를 시도합니다. 빨리 감기가 가능한 상황이기 때문에 **git merge** 명령처럼 빨리 감기 병합을 하고 작업을 종료합니다.
5. **git push** 명령으로 \[main\] 브랜치를 원격에 푸시합니다.
6. **git branch -d** 명령으로 필요 없어진 \[mybranch1\] 브랜치를 삭제합니다.

### tag

이번에는 CLI에서 커밋에 태그를 달아 보겠습니다. 이를 <font color="#00b050">태깅</font> 이라고 하는데요. 태그에는 사실 주석이 있는 태그와 간단한 태그 두 종류가 있습니다. 일반적으로 주석이 있는 태그의 사용을 권장합니다. 기본적으로 태그를 생성하는 명령은 **git tag** 입니다.

| git tag -a -m <간단한 메시지> <태그 이름> <브랜치 또는 체크섬> | -a 로 주석 있는(annotated) 태그를 생성합니다. 메시지와 태그 이름은 필수이며 브랜치 이름을 생략하면 HEAD에 태그를 생성합니다. |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| git push <원격 저장소 이름> <태그 이름>                 | 원격 저장소에 태그를 업로드합니다.                                                             |

그럼 CLI에서 태그를 작성해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline
5761085 (HEAD -> main, origin/main) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
991cb7e 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git tag -a -m "첫 번째 태그 생성" v0.1

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline
5761085 (HEAD -> main, tag: v0.1, origin/main) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
991cb7e 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git push origin v0.1
Enumerating objects: 1, done.
Counting objects: 100% (1/1), done.
Writing objects: 100% (1/1), 184 bytes | 184.00 KiB/s, done.
Total 1 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Ludobico/hello-git-cli.git
 * [new tag]         v0.1 -> v0.1
```

태그는 차후에 커밋을 식별할 수 있는 유용한 정보이므로 잘 활용하는 것이 좋습니다. 태그를 사용하면 GitHub의 \[Tags\] 탭에서 확인할 수 있고, \[Release\] 탭에서 다운받을 수 있다는 것도 기억하시길 바랍니다.

지금까지 CLI를 통해서 브랜치를 생성하고 브랜치 변경과 빨리 감기 병합을 했습니다. 그리고 빨리 감기가 가능한 상황에서 **rebase** 명령어는 **merge** 명령어와 같은 동작을 보인다는 것도 확인했습니다.

마지막으로 **tag** 명령어의 사용법에 대해서도 알아봤습니다. 태그는 정말 유용한 기능이니까 잘 배워서 활용해 보시길 바랍니다.

이제 조금 더 복잡한 상황에서 상황에서 병합과 리베이스를 알아볼 것입니다.

> switch 명령어와 restore 명령어
> Git을 공부하다 보면 checkout 명령어를 볼 수 있습니다. 이 명령어는 이전까지 자주 사용하던 명령인데 최근에는 잘 사용하지 않게 되었습니다. checkout 명령어의 기능을 크게 두 가지로 나누면 브랜치의 변경과 워킹트리의 파일 내용 복구입니다.
> 브랜치를 변경하는 명령은 switch 명령으로 대체되었고, 파일을 복구하는 명령은 restore 명령어로 대치되었습니다. 드물에 checkout 명령어가 필요한 경우도 있긴 하지만, 거의 사용할 필요가 없는 명령어가 되었습니다.
> restore 명령어는 유용한 기능을 많이 담고 있습니다. 이 책에서 따로 다루지는 않지만, Git Bash 창에 git restore --help 명령을 입력해 한번 살펴보세요

