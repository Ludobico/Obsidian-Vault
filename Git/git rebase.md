- [[#Rebase|Rebase]]
- [[#기본 명령어|기본 명령어]]
- [[#충돌 발생시|충돌 발생시]]
- [[#git rebase --continue 시 편집기(vim)가 뜰 때|git rebase --continue 시 편집기(vim)가 뜰 때]]
- [[#rebase 완료 후 로컬/원격이 "diverged" 상태가 되는 이유|rebase 완료 후 로컬/원격이 "diverged" 상태가 되는 이유]]
- [[#이 상태에서 일반 git push가 실패하는 이유|이 상태에서 일반 git push가 실패하는 이유]]
- [[#Pull 하라는 안내가 뜨면: "Don't pull" 선택|Pull 하라는 안내가 뜨면: "Don't pull" 선택]]
- [[#--force vs --force-with-lease|--force vs --force-with-lease]]
- [[#전체 흐름|전체 흐름]]

## Rebase

[[Git]]에서  내 브랜치의 커밋들을 다른 브랜치의 최신 지점 뒤로 재배치하는 작업입니다. **merge**가 두 히스토리를 합치는 새 커밋을 만드는 것과 달리, **rebase**는 커밋을 하나씩 다시 만들어(= 새 해시 부여) 히스토리를 일직선으로 만듭니다.

```
main:            A - B - C
feature/test:    A - B - D - E   (A에서 분기)
```

```bash
git checkout feature/test
git rebase main
```

```
결과:
main:            A - B - C
feature/test:    A - B - C - D' - E'   (D, E가 C 뒤로 재생되며 D', E'로 재작성됨)
```

D, E는 내용은 비슷하지만 **커밋 해시가 완전히 새로 부여**된다는 점이 핵심입니다. 이후 모든 문제가 여기서 파생됩니다.

## 기본 명령어

```bash
git switch feature/test
git rebase main
```

## 충돌 발생시

두 브랜치가 **같은 파일의 같은 부분을 다르게 고쳤다면** 커밋을 재생하는 도중 멈춥니다.

```bash
$ git rebase main
Auto-merging config.toml
CONFLICT (content): Merge conflict in config.toml
error: could not apply a1b2c3d... test 기능 추가
Resolve all conflicts manually, mark them as resolved with
"git add <conflicted_files>", then run "git rebase --continue".
```

<font color="#ffff00">config.toml</font>을 열어보면 다음과 같습니다.

```
<<<<<<< HEAD
value = "from-main"
=======
value = "from-feature"
>>>>>>> a1b2c3d (test 기능 추가)
```

세 마커(<<<<<<<, \=\=\=\=\=\==, >>>>>>>)를 직접 지우고 원하는 내용으로 정리한 뒤 다음을 실행합니다.

```bash
git add config.toml
git rebase --continue
```

<font color="#ffff00">package-lock.json</font>,<font color="#ffff00"> uv.lock</font>처럼 다른 설정 파일(pyproject.toml, package.json 등)로부터 자동 생성되는 파일이 충돌 마커를 포함한 채로 남으면, 손으로 줄 단위 병합을 시도하지 말고 원본 설정 파일을 먼저 정리한 뒤 재생성하는 것이 안전합니다.

## git rebase --continue 시 편집기(vim)가 뜰 때

충돌 해결 후 재커밋하면서 원래 커밋 메시지를 편집기로 보여줍니다. 메시지를 바꿀 필요가 없다면 그대로 저장·종료하면 됩니다.

```
:wq
```

## rebase 완료 후 로컬/원격이 "diverged" 상태가 되는 이유

```
Your branch and 'origin/feature/test' have diverged,
and have 3 and 2 different commits each, respectively.
```

- 앞 숫자(3): 로컬 브랜치에만 있는 커밋 수입니다 (main에서 새로 들어온 커밋 + rebase로 새 해시를 받은 내 커밋들).
- 뒤 숫자(2): origin에만 있는 커밋 수입니다 (rebase 하기 전의 옛 커밋들. 내용은 비슷해도 해시가 달라졌으니 git은 "다른 커밋"으로 취급합니다).

즉 코드가 실제로 어긋난 것이 아니라, **rebase가 커밋 해시를 재작성했기 때문에 생기는 정상적인 결과**입니다.

## 이 상태에서 일반 git push가 실패하는 이유

```bash
$ git push origin feature/test
 ! [rejected]        feature/test -> feature/test (non-fast-forward)
error: failed to push some refs to '...'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart.
```

일반 push는 **fast-forward**(원격 히스토리가 로컬 히스토리에 그대로 포함된 경우)만 허용합니다. rebase로 커밋 해시가 바뀌어 두 히스토리가 갈라졌으니, git은 "이대로 덮어써도 되는지" 확신할 수 없어 거부합니다.

## Pull 하라는 안내가 뜨면: "Don't pull" 선택

Git GUI 도구가 다음처럼 물어볼 수 있습니다.

```
It looks like the current branch "feature/test" might have been rebased.
Are you sure you still want to pull into it?
```

여기서 pull(= fetch + merge/rebase)을 하면, 원격에 남아있는 **rebase 이전의 옛 커밋**들이 로컬에 다시 합쳐지면서 방금 한 rebase 작업이 무의미해집니다. Don't pull을 선택하고, 대신 push로 원격을 갱신해야 합니다.

## --force vs --force-with-lease

```bash
git push --force-with-lease origin feature/test
```

| 옵션                 | 동작                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------ |
| --force            | 원격 상태를 확인하지 않고 무조건 덮어씁니다. 그 사이 다른 사람이 push한 커밋이 있어도 통째로 날아갈 수 있습니다.                  |
| --force-with-lease | 마지막으로 내가 fetch한 시점 이후 원격이 바뀌지 않았을 때만 덮어씁니다. 바뀌었다면 push를 거부해 남의 작업을 실수로 지우는 것을 방지합니다. |
특별한 이유가 없다면 항상 **--force-with-lease**를 쓰는 것이 안전합니다.

## 전체 흐름

```bash
git switch feature/test
git rebase main
# 충돌 발생 시:
#   1) 충돌 파일 직접 수정 (마커 제거)
#   2) 자동 생성 파일(lock 등)은 재생성
#   3) git add <파일>
#   4) git rebase --continue (편집기 뜨면 :wq)
# 모든 커밋 재생 완료 후:
git push --force-with-lease origin feature/test
```

