> 3-way merge
> 두 개의 브랜치를 병합할 때 사용되는 방식 중 하나로, 세 개의 커밋을 참조하여 병합을 수행하는 방법을 의미합니다.

### emergency bug handling procedure

갑작스레 버그를 발견한 상황을 생각해 보겠습니다. 보통 이 경우 하나 이상의 브랜치로 다른 기능 개선을 하고 있을 것입니다. 이런 상황에서 버그 수정은 다음과 같은 단계로 이루어집니다.

> 1. (옵션) 오류가 없는 버전(주로 Tag가 있는 커밋)으로 롤백합니다.
> 2. [main] 브랜치로부터 [hotfix] 브랜치를 생성합니다.
> 3. 빠르게 소스 코드를 수정하고 테스트를 완료합니다.
> 4. [main] 브랜치로 빨리 감기 병합 및 배포합니다.
> 5. 개발 중인 브랜치에도 병합합니다.

버그가 발생한 상황에서는 원래 작업 중이던 브랜치도 \[main\] 브랜치로부터 시작했으므로 같은 버그를 가지고 있을 것입니다. 그래서 \[hotfix\] 브랜치의 내용은 \[main\] 브랜치와 개발 브랜치 모두에 병합되어야 합니다. 보통 \[main\] 브랜치의 병합은 빨리 감기이기 때문에 쉽게 되는 반면, 개발 중인 브랜치의 병합은 병합 커밋이 생성되고, 충돌이 일어날 가능성이 높습니다.

이러한 상황을 가정하고 실습을 해 보겠습니다. 먼저 \[feature1\] 브랜치를 만들고 커밋을 하나 생성합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git switch -C feature1
Switched to a new branch 'feature1'

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ echo "기능 1 추가" >> file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git add file1.txt
warning: in the working copy of 'file1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git commit
[feature1 2e7b43a] 새로운 기능1 추가
 1 file changed, 1 insertion(+)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git log --oneline --graph --all -n2
* 2e7b43a (HEAD -> feature1) 새로운 기능1 추가
* 5761085 (tag: v0.1, origin/main, main) mybranch1 두 번째 커밋
```

지금 이 시점에서 장애가 발생했습니다. 그나마 다행인 점은 이미 커밋을 한 상태에서 장애가 발생했다는 점입니다. 현실에서는 커밋을 하기 모호한 상황에서 장애가 발생하게 됩니다. 이럴 때는 스태시(stash)를 사용할 수 있지만 **stash** 명령에 대해서는 뒷 장에서 설명할테니 일단 커밋을 한 직후에 장애가 발생했다고 가정합니다.

이제 버그를 고치기 위해 \[main\] 브랜치에서 \[hotfix\] 브랜치를 먼저 만들어야 합니다. 그리고 버그를 고쳐 커밋한 후 \[hotfix\] 브랜치를 \[main\] 브랜치에 병합합니다. 이렇게 하면 \[main\] 브랜치의 최신 커밋을 기반으로 \[hotfix\] 브랜치 작업을 했기 때문에 빨리 감기 병합이 가능한 상황입니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git switch -c hotfix main
Switched to a new branch 'hotfix'

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (hotfix)
$ git log --oneline --all -n2
2e7b43a (feature1) 새로운 기능1 추가
5761085 (HEAD -> hotfix, tag: v0.1, origin/main, main) mybranch1 두 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (hotfix)
$ echo "some hot fix" >> file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (hotfix)
$ git add file1.txt
warning: in the working copy of 'file1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (hotfix)
$ git commit
[hotfix 26928fc] hotfix 실습
 1 file changed, 1 insertion(+)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (hotfix)
$ git log --oneline -n1
26928fc (HEAD -> hotfix) hotfix 실습

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (hotfix)
$ git switch main
Switched to branch 'main'
Your branch is up to date with 'origin/main'.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git merge hotfix
Updating 5761085..26928fc
Fast-forward
 file1.txt | 1 +
 1 file changed, 1 insertion(+)

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git push
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 16 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 306 bytes | 306.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Ludobico/hello-git-cli.git
   5761085..26928fc  main -> main
```

아직 추가 작업이 남아 있습니다. 물론 긴급한 작업은 끝났으니 한시름 놓은 상태입니다. hotfix의 커밋은 버그 수정이었기 때문에 이 내용은 현재 개발 중인 \[feature1\] 브랜치에도 반영해야 합니다. 그런데 \[feature1\] 브랜치와 \[main\] 브랜치는 아래 그림처럼 서로 다른 분기로 진행되고 있습니다. 이 경우에는 빨리 감기 병합이 불가능하므로 <font color="#00b050">3-way 병합</font>을 해야 합니다.

![[Pasted image 20240906150600.png]]

게다가 모든 3-way 병합이 충돌을 일으키는 것은 아닙니다만 이번 실습에서는 고의적으로 두 브랜치 모두 `file1.txt` 를 수정했기 때문에 충돌이 발생합니다.

| main 브랜치의 file1.txt                                                | feature1 브랜치의 file1.txt                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------------- |
| hello git<br>second<br>third - my branch<br>fourth<br>some hot fix | hello git<br>second<br>third - my branch<br>fourth<br>기능 1 추가 |

일단 3-way 병합을 해 봅시다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git switch feature1
Switched to branch 'feature1'

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git log --oneline --all
26928fc (origin/main, main, hotfix) hotfix 실습
2e7b43a (HEAD -> feature1) 새로운 기능1 추가
5761085 (tag: v0.1) mybranch1 두 번째 커밋
d424559 mybranch1 첫 번째 커밋
991cb7e 두 번째 커밋
8217549 첫 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git merge main
Auto-merging file1.txt
CONFLICT (content): Merge conflict in file1.txt
Automatic merge failed; fix conflicts and then commit the result.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|MERGING)
$ git status
On branch feature1
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)
        both modified:   file1.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

1. **git merge main** 명령이 충돌로 인해 실패합니다.
2. **git status** 명령을 실행하여 충돌 대상 파일을 확인할 수 있습니다. 결과 메시지에서 볼 수 있는 것처럼 **git merge --abort** 명령을 통해 취소할 수도 있습니다.

여기에서 VS 코드를 열면 충돌 부분이 <font color="#00b050">다른 색으로</font> <font color="#00b0f0">표시되고</font> 위 쪽에는 흐릿한 글씨로 4개의 선택 메뉴가 보입니다.

- Accept Current Change : HEAD의 내용만 선택
- Accept Incoming Change : incoming 브랜치 내용만 선택
- Accept Both Change : 둘 다 선택
- Compare Changes : 변경 사항 비교

여기서는 둘 다 필요한 내용이므로 Accept Both Change를 선택합니다.

```txt
hello git
second
third - my branch
fouth - my branch
기능 1 추가
some hot fix
```

이제 변경 내용을 저장하고 다시 스테이지에 추가 및 커밋을 하면 수동 3-way 병합이 완료됩니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|MERGING)
$ cat file1.txt
hello git
second
third - my branch
fouth - my branch
기능 1 추가
some hot fix

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|MERGING)
$ git add file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|MERGING)
$ git status
On branch feature1
All conflicts fixed but you are still merging.
  (use "git commit" to conclude merge)

Changes to be committed:
        modified:   file1.txt


Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|MERGING)
$ git commit
[feature1 4a35c4c] 병합 커밋 생성

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git log --oneline --all --graph -n4
*   4a35c4c (HEAD -> feature1) 병합 커밋 생성
|\
| * 26928fc (origin/main, main, hotfix) hotfix 실습
* | 2e7b43a 새로운 기능1 추가
|/
* 5761085 (tag: v0.1) mybranch1 두 번째 커밋
```

1. **git add** 및 **git status** 명령을 수행하면 충돌한 파일의 수정을 완료한 후에 **git commit** 명령을 수행하면 된다는 것을 알 수 있습니다.
2. **git commit** 명령으로 충돌난 3-way 병합을 마무리 짓겠습니다.

### Clean up the tree by rebasing commits

3-way 병합을 하면 병합 커밋이 생성되기 때문에 트리가 다소 지저분해진다는 단점이 있습니다. 이럴 때 <font color="#ffff00">트리를 깔끔하게</font> 하고 싶다면 **git rebase** 명령을 사용할 수 있습니다. rebase는 내 브랜치의 커밋들을 재배치하는 것을 끝합니다.

리베이스의 원리를 살펴보면 다음과 같습니다.

> 1. HEAD와 대상 브랜치의 공통 조상을 찾습니다.
> 2. 공통 조상 이후에 생성한 커밋들(C4, C5 커밋)을 대상 브랜치 뒤로 재배치합니다.

![[Pasted image 20240906153959.png]]
<feature1 브랜치 rebase 전>

![[Pasted image 20240906154106.png]]
<feature1 브랜치 rebase 후>

먼저 \[feature1\] 브랜치는 HEAD 이므로 \* 이 붙어있습니다. 여기서 **git rebase main** 명령을 수행하면 공통 조상인 `C2` 이후의 커밋인 `C4` 와 `C5` 를 \[main\] 브랜치의 최신 커밋인 `C3` 의 뒤 쪽으로 재배치합니다. 그런데 재배치된 C4와 C5는 각각 `C4'` 와 `C5'` 가 되었습니다. 이 말은 <font color="#ffff00">리베이스된 커밋은 원래의 커밋과 다른 커밋이라는 뜻</font>입니다. 실습을 할 때도 리베이스 전과 후에 커밋 체크섬을 확인해 보면 값이 달라진 것을 직접 확인해 볼 수 있습니다.

> 리베이스는 어떤 경우에 사용해야 하나요?
> rebase 명령어는 주로 로컬 브랜치를 깔끔하게 정리하고 싶을 때 사용합니다. 원격에 푸시한 브랜치를 리베이스할 때는 조심해야 합니다. 여러 Git 가이드에서도 원격 저장소에 존재하는 브랜치에 대해서는 리베이스를 하지 말 것을 권합니다.

앞 절에서 만들었던 병합 커밋을 **git reset --hard** 명령으로 한 단계 되돌리고, **git rebase** 명령으로 커밋을 재배치해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git reset --hard HEAD~
HEAD is now at 2e7b43a 새로운 기능1 추가

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git log --oneline --graph --all -n3
* 26928fc (origin/main, main, hotfix) hotfix 실습
| * 2e7b43a (HEAD -> feature1) 새로운 기능1 추가
|/  
* 5761085 (tag: v0.1) mybranch1 두 번째 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git rebase main
Auto-merging file1.txt
CONFLICT (content): Merge conflict in file1.txt
error: could not apply 2e7b43a... 새로운 기능1 추가
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
hint: You can instead skip this commit: run "git rebase --skip".
hint: To abort and get back to the state before "git rebase", run "git rebase --abort".
Could not apply 2e7b43a... 새로운 기능1 추가
```

1. HEAD를 \[feature1\] 브랜치로 전환합니다.
2. **git reset --hard HEAD~** 명령으로 커밋을 한 단게 이전으로 되돌립니다. 이렇게 하면 병합커밋이 사라집니다.
3. 로그를 통해 커밋 체크섬을 확인합니다. 재배치 대상 커밋의 체크섬 값이 `2e7b43a` 라는 것을 알 수 있습니다.
4. 리베이스를 시도하지만 병합을 시도했을 때와 마찬가지로 <font color="#ffff00">충돌로 인해 리베이스는 실패</font>합니다. 여기서 실패 메시지를 잘 보면 수동으로 충돌을 해결한 후에 스테이지에 추가를 할 것을 알려줍니다. 또한 **git rebase --continue** 명령을 수행하라는 것도 알려줍니다.

다시 충돌을 해결하고 리베이스를 계속해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|REBASE 1/1)
$ git status
interactive rebase in progress; onto 26928fc
Last command done (1 command done):
   pick 2e7b43a 새로운 기능1 추가
No commands remaining.
You are currently rebasing branch 'feature1' on '26928fc'.
  (all conflicts fixed: run "git rebase --continue")

Changes not staged for commit:
applying : 새로운 기능1 추가
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   file1.txt

no changes added to commit (use "git add" and/or "git commit -a")

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|REBASE 1/1)
$ git add file1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|REBASE 1/1)
$ git status
interactive rebase in progress; onto 26928fc
Last command done (1 command done):
   pick 2e7b43a 새로운 기능1 추가
No commands remaining.
You are currently rebasing branch 'feature1' on '26928fc'.
  (all conflicts fixed: run "git rebase --continue")

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   file1.txt


Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1|REBASE 1/1)
$ git rebase --continue
[detached HEAD 96bbb64] applying : 새로운 기능1 추가
 1 file changed, 1 insertion(+)
Successfully rebased and updated refs/heads/feature1.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git log --oneline --graph --all -n2
* 96bbb64 (HEAD -> feature1) applying : 새로운 기능1 추가
* 26928fc (origin/main, main, hotfix) hotfix 실습

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (feature1)
$ git switch main
Switched to branch 'main'
Your branch is up to date with 'origin/main'.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git merge feature1
Updating 26928fc..96bbb64
Fast-forward
 file1.txt | 1 +
 1 file changed, 1 insertion(+)
```

1. 충돌 파일을 확인하고 이전과 같은 방식으로 vs 코드를 이용해서 수동으로 파일 내용을 수정하 저장합니다.
2. 스테이지에 변경된 파일을 추가합니다.
3. **git rebase --continue** 명령을 수행해서 이어서 리베이스 작업을 진행합니다. 여기서 차이점은 **git merge** 명령은 마지막 단계에서 **git commit** 명령을 사용해서 병합 커밋을 생성한 후 마무리 하지만, git rebase 명령은 git rebase --continue 명령을 사용해야 합니다.
4. 로그를 확인합니다. 병합과는 달리 병합 커밋도 없고 히스토리도 한 줄로 깔끔해졌습니다. 또한 \[feature1\] 브랜치가 가리키는 커밋의 체크섬 값이 `96bbb64` 으로 바뀐 것을 볼 수 있습니다. 이는 앞서 설명한 컷처럼 리베이스를 하면 커밋 객체가 바뀌기 때문입니다.
5. 마지막으로 \[main\] 브랜치에서 \[feature1\] 브랜치로 병합합니다. 한 줄이 되었기 때문에 빨리감기 병합을 수행합니다.

리베이스와 병합의 마지막 단계에서 명령어가 다른 것이 이상하다고 여길 수 있는데요, 3-way 병합은 기존 커밋의 변경 없이 새로운 병합 커밋을 하나 생성합니다. 따라서 충돌도 한 번만 발생합니다. 충돌 수정 완료 후 **git commit** 명령을 수행하면 병합 작업이 완료되는 것이죠

그러나 리베이스는 재배치 대상 커밋이 여러 개일 경우 여러 번 충돌이 발생할 수 있습니다. 또한 기존의 커밋을 하나씩 단계별로 수정하기 때문에 **git rebase --continue** 명령으로 충돌로 인해 중단된 **git rebase** 명령을 재개하게 됩니다. 여러 커밋에 충돌이 발생했다면 충돌을 해결할 때마다 git rebase --continue 명령을 매번 입력해야 합니다. 복잡해지고 귀찮기 때문에 이런 경우에는 병합을 수행하는 것이 더 간단할 수도 있습니다.

|     | 3-way 병합     | 리베이스                        |
| --- | ------------ | --------------------------- |
| 특징  | 병합 커밋 생성     | 현재 커밋들을 수정하면서 대상 브랜치 위로 재배치 |
| 장점  | 한 번만 충돌 발생   | 깔끔한 히스토리                    |
| 단점  | 트리가 약간 지저분해짐 | 여러 번 충돌이 발생할 수 있음           |

![[Pasted image 20240909113423.png]]

### Tree prune

보통 한 PC에서 커밋을 만들고 푸시했는데, 다른 PC에서 또 다른 커밋을 하게 되면 이전 커밋을 부모로 한 커밋이 생깁니다. 그 상황에서 뒤늦게 풀을 시도하면 자동으로 3-way 병합이 되기 때문에 아래 그림 같은 모양이 되는 것입니다.

![[Pasted image 20240909113920.png]]

지금 같은 경우는 불필요하게 병합 커밋이 생긴 상황입니다. 이 상황을 해결하려면

**reset --hard** 명령으로 병합 커밋을 되돌리고 **git rebase** 명령을 사용하는 것입니다.

먼저 가지 커밋을 하나 만들어 보겠습니다.가지를 만들기 위해 정상인 커밋을 만들고 푸시합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ echo "main1" > main1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git add main1.txt 
warning: in the working copy of 'main1.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git commit -m "main 커밋 1"
[main edcbbc8] main 커밋 1
 1 file changed, 1 insertion(+)
 create mode 100644 main1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git push origin main
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Delta compression using up to 16 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 285 bytes | 285.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Ludobico/hello-git-cli.git
   26928fc..edcbbc8  main -> main

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline -n1
edcbbc8 (HEAD -> main, origin/main, origin/HEAD) main 커밋 1

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ ls
file1.txt  main1.txt
```

일단 평범하게 커밋을 하나 생성했습니다. 이제 **reset --hard** 명령을 이용해서 한 단계 이전 커밋으로 이동합니다. 여기에서 다시 커밋을 생성하면 가지가 하나 생겨날 것입니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git reset --hard HEAD~
HEAD is now at 26928fc hotfix 실습

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ echo "main2" >main2.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git add .
warning: in the working copy of 'main2.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git commit -m "main2 커밋"
[main 637bfee] main2 커밋
 1 file changed, 1 insertion(+)
 create mode 100644 main2.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --graph --all -n3
* 637bfee (HEAD -> main) main2 커밋
| * edcbbc8 (origin/main, origin/HEAD) main 커밋 1
|/  
* 26928fc hotfix 실습
```

1. hard reset 모드로 \[main\] 브랜치를 한 단계 되돌립니다.
2. `main2.txt` 파일을 생성하고 커밋을 합니다.
3. 로그를 확인해 보면 main1 커밋과 main2 커밋 모두 `26928fc` 커밋을 부모로 하는 커밋이므로 가가 생긴 것을 알 수 있습니다.

지금 상황에서 풀을 하면 어떻게 될까요? **git pull** 명령은 **git fetch + git merge** 이기때문에 가지를 병합하기 위해서 병합 커밋이 생기고 괜히 커밋 히스토리가 지저분해집니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git pull
Merge made by the 'ort' strategy.
 main1.txt | 1 +
 1 file changed, 1 insertion(+)
 create mode 100644 main1.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --graph --all -n4
*   78f4bc0 (HEAD -> main) Merge branch 'main' of https://github.com/Ludobico/hello-git-cli
|\  
| * edcbbc8 (origin/main, origin/HEAD) main 커밋 1
* | 637bfee main2 커밋
|/  
* 26928fc hotfix 실습
```

1. **git pull** 명령을 수행합니다. 자동으로 병합 커밋이 생성됩니다. 
2. 로그를 확인해 보면 병합 커밋이 생성된 것을 알 수 있습니다.

병합 커밋이 생성되면 그때 hard reset 모드를 이용해 커밋을 되돌리고 재배치하면 됩니다. 이제 병합 커밋을 되돌린 후에 **git rebase** 명령으로 가지를 없애 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git reset --hard HEAD~
HEAD is now at 637bfee main2 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git rebase origin/main
Successfully rebased and updated refs/heads/main.

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git log --oneline --all --graph -n3
* cb7ddd8 (HEAD -> main) main2 커밋
* edcbbc8 (origin/main, origin/HEAD) main 커밋 1
* 26928fc hotfix 실습

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/hello-git-cli (main)
$ git push
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Delta compression using up to 16 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 314 bytes | 314.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Ludobico/hello-git-cli.git
   edcbbc8..cb7ddd8  main -> main
```

1. **reset --hard HEAD~** 명령을 이용해서 커밋을 하나 되돌립니다. 이 경우 마지막 커밋은 병합 커밋이었으므로 병합되기 전 커밋 `637bfee` 으로 돌아가게 됩니다. 이제 HEAD는 가지로 튀어나온커밋을 가리키고 있으므로 이 커밋을 재배치해야 합니다.
2. **git rebase origin/main** 명령을 수행하면 로컬 \[main\] 브랜치의 가지 커밋이 \[origin/main\] 브랜치 위로 재배치됩니다.
3. 로그를 확인하고 원격 저장소에 푸시합니다.

### Rebase cautions

리베이스할 때 중요한 주의사항이 있습니다. <font color="#ffff00">원격 저장소에 푸시한 브랜치는 리베이스하지 않는 것이 원칙</font>입니다. 예를 들어 C1 커밋을 원격에 풋하고 리베이스하게 되면 원격에는 C1이 존재하고 로컬에는 다른 커밋이 C1\` 이 생성됩니다. 이때 내가 아닌 다른 사용자는 원격에 있던 C1을 병합할 수 있습니다. 그런데 변경된 C1\` 도 언젠가는 원격에 푸시되고 그럼 원격에는 실상 같은 커밋이었던 C1과 C1\`이 동시에 존재하게 됩니다.

