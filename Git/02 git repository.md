- [[#push to repository|push to repository]]
- [[#Update the local repository with new commits from the remote repository|Update the local repository with new commits from the remote repository]]


Git Repository 는 [[Git]]이 관리하는 프로젝트의 저장소를 의미합니다. repository는 파일과 폴더 뿐만 아니라, 이들의 변경 이력, 커밋 기록, 브랜치, 태그 등을 포함하는 구조로 구성되어 있습니다.

## push to repository

![[Pasted image 20240903144325.png]]

Github에서 `git-test` 라는 레파지토리를 만들고, 이 원격 저장소 주소를 [[01 git commit]] 에서 만든 `git-test` 로컬 저장소에 알려 주고, 로컬 저장소에 만들었던 커밋들을 원격 저장소에 올려 보겠습니다.

로컬 저장소의 `git-test` 에서 **git remote add origin** 명령으로 로컬 저장소에 연결할 원격 저장소 주소를 알려 줍니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git remote add origin https://github.com/Ludobico/git-test.git
```

지금까지 만든 커밋을 둘 <font color="#ffff00">브랜치</font>를 짓겠습니다. 저장소 안에 `main` 이라는 이름의 브랜치를 만듭니다. 다음처럼 **git branch** 명령을 입력합니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (master)
$ git branch -M main
```

이제 로컬 저장소에 있는 커밋들을 **git push** 명령으로 원격 저장소에 올려 보겠습니다. 다음 명령은 원격 저장소(origin) 의 main 이라는 브랜치에 내 커밋들을 올려라(push)라는 뜻입니다.

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (main)
$ git push origin main
Enumerating objects: 6, done.
Counting objects: 100% (6/6), done.
Delta compression using up to 16 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (6/6), 536 bytes | 536.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Ludobico/git-test.git
 * [new branch]      main -> main
```

## Update the local repository with new commits from the remote repository

다음과 같이 **git pull origin main** 명령을 입력합니다. <font color="#ffff00">원격 저장소에 새로운 커밋이 있다면 그걸 내 로컬 저장소에 받아오라는 명령</font>입니다. 

![[Pasted image 20240903151629.png]]

![[Pasted image 20240903151647.png]]

```bash
Ludobico@Ludobico MINGW64 ~/OneDrive/Desktop/repoSub/git-test (main)
$ git pull origin main
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
Unpacking objects: 100% (3/3), 928 bytes | 40.00 KiB/s, done.
From https://github.com/Ludobico/git-test
 * branch            main       -> FETCH_HEAD
   a476c66..04d7490  main       -> origin/main
Updating a476c66..04d7490
Fast-forward
 README.md | 3 +++
 1 file changed, 3 insertions(+)
```

