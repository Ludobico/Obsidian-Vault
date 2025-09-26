> HEAD

- 현재 checkout된 브랜치의 가장 최근 커밋을 가리킵니다.

> HEAD~1

- HEAD에서 한 단계 이전 커밋을 가리킵니다.
- `git reset HEAD~1` -> 마지막 커밋 취소

> HEAD^ 또는 HEAD^1

기본적으로 **첫 번째 부모 커밋**을 가리킵니다.

---

커밋을 되돌리면서 staged 상태까지 해제하고 싶으면

```bash
git reset HEAD~1   # --mixed 기본값 사용
```

커밋만 되돌리고 staged 상태는 유지하고 싶으면

```bash
git reset --soft HEAD~1
```

