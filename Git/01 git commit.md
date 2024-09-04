- [[#Build first commit|Build first commit]]
- [[#Move to another commit|Move to another commit]]


`git commit` 은 [[Git]] 에서 **파일의 변경 사항을 로컬 저장소에 저장하는 명령어**입니다. Git은 분산 버전 관리 시스템으로, 프로젝트의 모든 변경 사항을 추적하고, 각 변경 사항을 커밋(commit) 이라는 단위로 기록하여 관리합니다. `git commit` 명령어는 소스 코드의 특정 상태를 기록하고, 이후 해당 상태로 되돌리거나 협업을 위해 공유할 수 있도록 합니다.

## Build first commit

환경은 다음과 같은 환경에서 실행합니다. `git-test` 라는 폴더안에 `README.md` 파일이 있습니다.

```bash
 C:\Users\aqs45\OneDrive\Desktop\repoSub\git-test 디렉터리

2024-09-03  오후 02:12    <DIR>          .
2024-09-03  오후 02:12    <DIR>          ..
2024-09-03  오후 02:12                17 README.md
```

```markdown
git commit test
```

여기에서 **git init** 이라는 명령어를 사용해 보겠습니다. git init 이라는 명령을 실행하면 `.git` 폴더가 생성됩니다. .git 에는 Git으로 <font color="#ffff00">생성한 버전들의 정보와 원격 주소 등</font>이 들어있고, 이 `.git` 폴더를 우리는 <font color="#ffff00">로컬 저장소</font>라고 부릅니다.

`Initialized empty Git repositor` 라는 텍스트가 나오면 성공입니다.

```bash
(llm_architecture) C:\Users\aqs45\OneDrive\Desktop\repoSub\git-test>git init
Initialized empty Git repository in C:/Users/aqs45/OneDrive/Desktop/repoSub/git-test/.git/
```

일반 프로젝트 폴더에 **git init** 명령(Git 초기화 과정이라고도 합니다.)을 통해 로컬 저장소를 만들면 그때부터 이 폴더에서 저장관리를 할 수 있습니다.

앞에서 생성한 `README.md`  파일을 하나의 버전으로 만들어 보겠습니다. Git 에서는 이렇게 생성된 각 버전을 <font color="#ffff00">커밋(commit)</font> 이라고 부르고, 이를 커밋한다 라고 표현하기도 합니다.

먼저 버전 관리를 위해 내 정보를 등록해야 합니다. `git bash` 또는 `terminal` 창에서 GitHub에 등록한 이메일 주소와 username을 큰 따옴표로 묶어 입력합니다.

```bash
git config --global user.email "gittest@gmail.com"

git config --global user.name "gittest"
```

그다음에는 버전으로 만들 파일을 선택합니다. 조금 전 만들어놓은 `README.md` 파일을 해 보겠습니다. 다음과 같이 **git add** 명령을 입력합니다.

```bash
(llm_architecture) C:\Users\aqs45\OneDrive\Desktop\repoSub\git-test>git add README.md
```

커밋에는 상세 설명을 적을 수 있습니다. 설명을 잘 적어 놓으면 내가 이 파일을 왜 만들었는지, 왜 수정했는지 알 수 있고, 해당 버전을 찾아 그 버전으로 코드를 바꿔 시간 여행을 하기도 수월합니다. **git commit** 명령을 입력하여 첫 번째 버전을 만들어보겠습니다.

```bash
$ git commit -m "사이트 설명 추가"
[master (root-commit) c18eb60] 사이트 설명 추가
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

**-m** 은 message의 약자입니다.

이번에는 `README.md` 파일을 수정하고 두 번째 커밋을 만들어 보겠습니다. 파일을 열어 다음과 같이 수정합니다.

```markdown
git commit test test
```

git add 명령으로 `README.md` 파일을 수정하고 설명 업데이트라는 설명을 붙여서 git commit 명령을 실행하면 커밋이 만들어집니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git add README.md 

Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git commit -m "설명 업데이트"
[master a476c66] 설명 업데이트
 1 file changed, 1 insertion(+), 1 deletion(-)
```

## Move to another commit

이렇게 만들어 둔 커밋으로 우리는 언제든 시간 여행을 할 수 있습니다. 개발을 하다가 요구 사항을 바뀌어서 이전 커밋부터 다시 개발하고 싶다면 Git을 사용해 그 커밋으로 돌아가면 됩니다.

먼저 **git log** 명령으로 지금까지 만든 커밋을 확인합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git log
commit a476c6688d3821a9cb4aa37dc293da900a7bc067 (HEAD -> master)
Author: Ludobico <89598307+Ludobico@users.noreply.github.com>
Date:   Tue Sep 3 14:25:27 2024 +0900

    설명 업데이트

commit c18eb60b19429a03148b88e102cc4fbe886945e2
Author: Ludobico <89598307+Ludobico@users.noreply.github.com>
Date:   Tue Sep 3 14:22:13 2024 +0900

    사이트 설명 추가
```
git log 명령은 최신 커밋부터 보여줍니다.

우리가 되돌리려는 커밋은 첫 번째 커밋이니 <font color="#ffff00">커밋 아이디의 앞 7자리를 복사</font> 하고 **git checkout** 명령으로 해당 커밋으로 코드를 되돌립니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git checkout c18eb60
Note: switching to 'c18eb60'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by switching back to a branch.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -c with the switch command. Example:

  git switch -c <new-branch-name>

Or undo this operation with:

  git switch -

Turn off this advice by setting config variable advice.detachedHead to false

HEAD is now at c18eb60 사이트 설명 추가
```
당연히 <font color="#ffff00">아이디 전체를 복사</font>해도 되지만 간편하게 앞 7자리만 복사하는 것입니다.

`README.md` 를 보시면 이전 커밋으로 돌아간 걸 확인할 수 있습니다.

```md
git commit test
```

다시 git checkout 명령으로 <font color="#ffff00">최신 커밋인 두 번째 커밋으로 돌아가겠습니다</font>.  위와 같은 방법으로 두 번째 커밋 아이디인 git checkout a476c6688를 입력해도 되지만, 간단하게 **git checkout -** 만 입력해도 됩니다.

```bash
$ git checkout -
Previous HEAD position was c18eb60 사이트 설명 추가
Switched to branch 'master'
```

이 처럼 **checkout** 명령어를 사용해서 원하는 커밋 시점으로 파일을 되돌릴 수 있습니다. 이 책에서는 이를 체크아웃한다 라고 표현합니다.

> checkout 명령어는 주의해서 사용하세요
> checkout 명령어는 Git에서 오래 전부터 지원하는 명령어인데, 너무 많은 기능을 포함하고 있습니. 그래서 최근 checkout 명령어의 주요 기능이 switch 명령어와 restore 명령어로 나누어졌습니다. switch는 브랜치 간 이동하는 명령어이고, restore는 커밋에서 파일들을 복구하는 명령어입니다.


