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

지금 이 시점에서 장애가 발생했습니다. 그나마 다행인 점은 이미 커밋을 