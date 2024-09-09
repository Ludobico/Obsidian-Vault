- [[#Inner workings of git add|Inner workings of git add]]
	- [[#Inner workings of git add#Create local repository|Create local repository]]
	- [[#Inner workings of git add#Check the file state|Check the file state]]
- [[#Ineer workings of git commit|Ineer workings of git commit]]
	- [[#Ineer workings of git commit#Tree object|Tree object]]
- [[#Dive into commit objects|Dive into commit objects]]
	- [[#Dive into commit objects#file update and add|file update and add]]

## Inner workings of git add

### Create local repository

먼저 명령을 실습할 로컬 저장소를 만들어 줍니다. \[git-test\] 이름의 폴더를 만든 후 **git init** 명령으로 로컬 저장소를 생성합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub
$ mkdir git-test

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub
$ cd git-test/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test
$ git init
Initialized empty Git repository in C:/Users/aqs45/OneDrive/Desktop/repoSub/git-test/.git/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ ls -al
total 8
drwxr-xr-x 1 Ludobico 197121 0 Sep  9 13:43 ./
drwxr-xr-x 1 Ludobico 197121 0 Sep  9 13:43 ../
drwxr-xr-x 1 Ludobico 197121 0 Sep  9 13:43 .git/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ ls -al .git
total 11
drwxr-xr-x 1 Ludobico 197121   0 Sep  9 13:43 ./
drwxr-xr-x 1 Ludobico 197121   0 Sep  9 13:43 ../
-rw-r--r-- 1 Ludobico 197121 130 Sep  9 13:43 config
-rw-r--r-- 1 Ludobico 197121  73 Sep  9 13:43 description
-rw-r--r-- 1 Ludobico 197121  23 Sep  9 13:43 HEAD
drwxr-xr-x 1 Ludobico 197121   0 Sep  9 13:43 hooks/
drwxr-xr-x 1 Ludobico 197121   0 Sep  9 13:43 info/
drwxr-xr-x 1 Ludobico 197121   0 Sep  9 13:43 objects/
drwxr-xr-x 1 Ludobico 197121   0 Sep  9 13:43 refs/
```

1. 먼저 비어있는 폴더를 git-test 라는 이름으로 생성합니다.
2. **git init** 명령을 수행하면 현재 폴더 아래에 \[.git\] 폴더가 생성됩니다. 이 폴더가 로컬 저장소입니다.
3. **ls -al** 명령으로 \[.git\] 폴더 내부를 확인해 보면 다양한 폴더가 생성되어 있다는 것을 알 수 있습니다.

동작 원리를 살펴볼 수 있는 명령은 다음과 같습니다. 이 명령들은 저수준(low-level) 명령어라고 하는데 실제로 많이 사용되는 명령은 아닙니다. 이번 실습에서 동작 원리를 이해하기 위해서 사용하는 명령이라고 생각하면 됩니다.

| git hash-object <파일명> | 일반 파일의 체크섬을 확인할 때 사용합니다.                                                        |
| --------------------- | ------------------------------------------------------------------------------- |
| git show <체크섬>        | 해당 체크섬을 가진 객체의 내용을 표시합니다.                                                       |
| git ls-files --stage  | 스테이지 파일의 내용을 표시합니다.<br>스테이지 파일은 git add 명령을 통해 생성되는데 .git/index 파일이 스테이지 파일입니다. |

### Check the file state

이제 cat-hanbit 이라는 텍스트를 담은 `cat.txt` 파일을 하나 생성한 후 **git status** 명령을 실행해 워킹트리의 상태를 확인해 보겠습니다. 

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ echo "cat-hanbit" > cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git status
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        cat.txt

nothing added to commit but untracked files present (use "git add" to track)
```

`cat.txt` 파일을 생성하고 **git status** 명령을 수행하면 바로 생성된 `cat.txt` 파일의 상태가 Untracked라는 것을 알 수 있습니다. **git status** 명령은 정확하게 어떤 일을 하는 걸까요? 바로 <font color="#ffff00">워킹트리와 스테이지, 그리고 HEAD 커밋 세 가지 저장 공간의 차이를 비교</font>해서 보여 줍니다.

![[Pasted image 20240909140052.png]]

이번에는 파일의 체크섬을 확인해 보겠습니다. **git hash-object <파일명>** 명령을 이용하면 해당 파일의 체크섬을 확인할 수 있습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git hash-object cat.txt
ff5bda20472c44e0b85e570185bc0769a6adec68
```

해시 체크섬은 같은 내용의 파일이라면 언제나 똑같은 값이 나옵니다. 값이 다르게 나왔다면 텍스트의 내용이 다른 것입니다.

이제 **git add** 명령으로 스테이지에 파일을 추가한 후 파일 상태를 확인해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git add cat.txt 
warning: in the working copy of 'cat.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git status
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   cat.txt


Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ cd .git

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ ls -a
./  ../  config  description  HEAD  hooks/  index  info/  objects/  refs/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ file index
index: Git index, version 2, 1 entries

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ git ls-files --stage
100644 ff5bda20472c44e0b85e570185bc0769a6adec68 0       cat.txt
```

1. 스테이지에 untracked 상태의 파일을 추가합니다.
2. **git status** 명령으로 상태를 확인합니다. cat.txt 가 스테이지에 추가된 것을 확인한 후에 \[.git\] 폴더를 보면 index 라는 파일이 생긴 것을 알 수 있습니다.
3. file 명령을 이용해 .git/index의 정체를 확인하면 Git index 라는 것을 알 수 있습니다. index는 스테이지의 다른 이름입니다. 즉, <font color="#ffff00">index 파일이 Git의 스테이지</font>입니다.
4. 스테이지의 파일 내용을 확인합니다. `cat.txt` 파일이 스테이지에 들어 있으며 체크섬은 좀 전에 확인한 값과 정확하게 일치한다는 것을 알 수 있습니다.

이 상태에서 \[.git\] 폴더를 조금 더 살펴보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ ls -a ./objects/
./  ../  ff/  info/  pack/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ ls -a ./objects/ff
./  ../  5bda20472c44e0b85e570185bc0769a6adec68

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ git show ff5bda
cat-hanbit
```

1.  \[.git/objects\] 폴더 아래에 ff/ 로 시작하는 폴더가 새로 생긴 것을 알 수 있습니다.
2. .git/objects/ff 폴더 아래에는 5dba로 시작하는 파일이 하나 있는데 폴더명과 파일명을 합쳐보면 `ff5dba` 입니다. 정확하게 앞에서 확인했던 체크섬 값입니다. \[objects\] 폴더 안에 존재하는 파일들은 Git 객체입니다.
3. **git show** 명령으로 ff5bda 객체의 내용을 확인하면 cat-hanbit 이라는 텍스트 파일이라는 것을 알 수 있습니다.

![[Pasted image 20240909140142.png]]

체크섬을 이용해서 객체의 종류와 내용을 확인할 수 있는 다른 명령으로 **git cat-file** 도 있습니다.

| git cat-fie -t <체크섬>       | 해당 체크섬을 가진 객체의 타입을 알려 주는 명령입니다.     |
| -------------------------- | ----------------------------------- |
| git cat-file <객체 타입> <체크섬> | 객체의 타입을 알고 있을 때 해당 파일의 내용을 표시해 줍니다. |

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ git cat-file -t ff5bda
blob

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test/.git (GIT_DIR!)
$ git cat-file blob ff5bda
cat-hanbit
```

1. `ff5bda` 객체가 `blob` 이라는 것을 알 수 있습니다. blob은 <font color="#00b050">binary large object</font> 의 줄임말로 <font color="#ffff00">스테지에 올라간 파일 객체는 blob</font>이 됩니다.
2. `ff5bda` 객체가 blob 이라는 것을 알았으므로 객체 내용을 확인하기 위해 **git cat-file blob** 명령을 사용합니다.

이번 절에서는 **git add** 명령이 워킹트리에 존재하는 파일을 스테이지에 추가하는 명령이라는 것을 직접 확인했습니다. 이때 해당 파일의 체크섬 값과 동일한 이름을 가지는 blob 객체가 생성되고 이 객체는 .git/objects 파일에 저장됩니다. 그리고 스테이지의 내용은 .git/index 에 기록됩니다.


## Ineer workings of git commit

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git commit
[master (root-commit) 9584654] 커밋 확인용 커밋
 1 file changed, 1 insertion(+)
 create mode 100644 cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git log
commit 9584654baecb151999d7c72aa636f8271f6bf649 (HEAD -> master)
Author: Ludobico <aqs450@gmail.com>
Date:   Mon Sep 9 14:08:45 2024 +0900

    커밋 확인용 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git status
On branch master
nothing to commit, working tree clean
```

로컬 저장소에서 최초의 커밋을 하나 만들었습니다. 그리고 **git status** 명령으로 상태 확인을 하면 "아무것도 없고 워킹트리는 깨끗하다" 라는 메시지를 볼 수 있습니다.

커밋을 하면 [[Git]] 에 어떤 변화가 일어나는지부터 살펴보겠습니다. 방금 한 커밋의 체크섬인 `9584654` 이 있는 폴더를 확인해보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ ls -a .git/objects/
./  ../  7a/  95/  ff/  info/  pack/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ ls -a .git/objects/95/
./  ../  84654baecb151999d7c72aa636f8271f6bf649

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git show 9584654
commit 9584654baecb151999d7c72aa636f8271f6bf649 (HEAD -> master)
Author: Ludobico <aqs450@gmail.com>
Date:   Mon Sep 9 14:08:45 2024 +0900

    커밋 확인용 커밋

diff --git a/cat.txt b/cat.txt
new file mode 100644
index 0000000..ff5bda2
--- /dev/null
+++ b/cat.txt
@@ -0,0 +1 @@
+cat-hanbit
```

1. 방금 만든 커밋 `9584654` 은 \[.git/objects/95\] 폴더 아래에 있습니다. 특이한 점은 모든 커밋은 체크섬이 다르기 때문에 예제와는 다른 체크섬을 가집니다.
2. **git show** 명령으로 `9584654` 객체를 보면 타입이 commit 입니다. 즉 커밋 객체라는 것을 재확인할 수 있습니다.

이 상태에서 스테이지의 내용을 다시 한번 확인해 보겠습니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git ls-files --stage
100644 ff5bda20472c44e0b85e570185bc0769a6adec68 0       cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git status
On branch master
nothing to commit, working tree clean
```

<font color="#ffff00">놀랍게도 스테이지는 비어 있지 않습니다.</font> **git status** 명령은 앞에서 워킹트리, 스테이지, HEAD 커밋 세 저장 공간을 비교한다고 했는데, 사실 <font color="#ffff00">clean의 뜻은 깨끗하다는 의미가 아닌 "워킹트리와 스테이지, 그리고 HEAD 커밋의 내용이 모두 똑같다" 라는 의미</font>입니다.

**git status**로 clearn한 상태는 "워킹트리 = 스테이지 = HEAD" 커밋이라는 점을 꼭 기억하기 바랍니다.

![[Pasted image 20240909142904.png]]

### Tree object

앞서 ls -a .git/objects/ 명령을 수행했을 때 blob과 커밋 말고도 `7a` 라는 이름의 또 다른 객체가 하나 더 있었던 것을 기억할 겁니다. 

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ ls -a .git/objects/
./  ../  7a/  95/  ff/  info/  pack/

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ ls -a .git/objects/7a
./  ../  5459aa5fe7865d499c6d1c0c7c7f8b278fb74f

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git show 7a5459
tree 7a5459

cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git ls-tree 7a5459
100644 blob ff5bda20472c44e0b85e570185bc0769a6adec68    cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git ls-files --stage
100644 ff5bda20472c44e0b85e570185bc0769a6adec68 0       cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git log --oneline -n1
9584654 (HEAD -> master) 커밋 확인용 커밋

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git cat-file -t 9584654
commit

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git cat-file commit 9584654
tree 7a5459aa5fe7865d499c6d1c0c7c7f8b278fb74f
author Ludobico <aqs450@gmail.com> 1725858525 +0900
committer Ludobico <aqs450@gmail.com> 1725858525 +0900

커밋 확인용 커밋
```

1. .git/objects/7a 폴더를 확인해 보면 7a5459 객체가 있는 것을 알 수 있습니다. 체크섬이 같은 객체는 같은 내용을 가지게 됩니다.
2. **git show** 명령으로 정체를 확인해 보면 타입이<font color="#00b050"> tree</font> 입니다. 이를 트리 객체라고 합니다.
3. **git ls-tree** 명령으로 트리 객체의 내용을 볼 수 있는데 이 내용은 예상한 것처럼 스테이지와 동일합니다.
4. 커밋 객체의 체크섬을 이용해 타입을 확인해 보면 <font color="#00b050">commit</font> 이라는 것을 알 수 있습니다.
5. `9584654` 커밋 객체의 내용을 들여다 보면 커밋 메시지와 트리 객체로 구성되어 있다는 것을 알 수 있습니다. 트리 객체의 체크섬은 1. 에서 확인한 객체와 일치합니다.

![[Pasted image 20240909143731.png]]

지금까지 내용을 요약하면 다음과 같습니다.

1. 커밋하면 스테이지의 객체로 트리 객체가 만들어집니다.
2. 커밋에는 커밋 메시지와 트리 객체가 포함됩니다.

## Dive into commit objects

앞서 **git commit** 명령의 동작 원리를 살펴보면 커밋이 객체라는 것을 알 수 있었습니다. 이번에는 파일을 수정해 스테이지에 추가해 보고, 트리 객체에 직접 커밋해보며 커밋 객체의 내부와 동작 방식을 살펴보겠습니다.

### file update and add

이번에는 파일의 내용을 수정하고 스테이지에 추가해 보겠습니다. 먼저 `cat.txt` 파일에 내용을 한 줄 추가해 수정한 후 현재 폴더와 스테이지 그리고 커밋 객체의 내용을 살펴봅니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ cat cat.txt 
cat-hanbit

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git hash-object cat.txt
ff5bda20472c44e0b85e570185bc0769a6adec68

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ echo "Hello, cat-hanbit" >> cat.txt 

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git hash-object cat.txt
f3e6fa5c881ffb692cf2f2353dc2e90ce5a207f8

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git ls-files --stage
100644 ff5bda20472c44e0b85e570185bc0769a6adec68 0       cat.txt

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git ls-tree HEAD
100644 blob ff5bda20472c44e0b85e570185bc0769a6adec68    cat.txt
```

아직 변경 사항을 스테이지에 추가하지 않았기 때문에 다음 그림처럼 현재 워킹트리의 체크섬만 다른 것을 확인할 수 있습니다. 변경된 파일은 modified 라고 했습니다.<font color="#ffff00"> modified는 스테이지와 워킹트리의 내용이 다른 파일을 일컫는 말</font>입니다. 

![[Pasted image 20240909144428.png]]

이제 변경 내용을 스테이지에 추가하면 스테이지에 있는 `cat.txt` 파일의 체크섬이 워킹트리의 내용과 같아집니다. 이 상태를 <font color="#00b050">staged 상태</font>라고합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git add cat.txt 
warning: in the working copy of 'cat.txt', LF will be replaced by CRLF the next time Git touches it

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git ls-files --stage
100644 f3e6fa5c881ffb692cf2f2353dc2e90ce5a207f8 0       cat.txt
```

![[Pasted image 20240909144645.png]]

다시 말해 staged 상태는 워킹트리와 스테이지의 내용은 같지만 HEAD 커밋과는 다른 상태를 말합니다.

| staged | working tree = stage $\ne$ HEAD |
| ------ | ------------------------------- |
| clean  | working tree = stage = HEAD     |

