- [[#전체 리포지토리 다운로드하기|전체 리포지토리 다운로드하기]]
	- [[#전체 리포지토리 다운로드하기#다운로드할 파일 선택하기|다운로드할 파일 선택하기]]
- [[#로컬 폴더에 파일 다운로드하기|로컬 폴더에 파일 다운로드하기]]


## 전체 리포지토리 다운로드하기

`snapshot_download()` 함수는 특정 버전의 전체 리포지토리를 다운로드합니다. 이 함수는 내부적으로 [[hf_hub_download]] 함수를 사용하므로, 다운로드한 모든 파일은 로컬 디스크에 캐시되어 저장됩니다. 다운로드는 여러 파일을 동시에 받아오기 때문에 빠르게 진행됩니다.

전체 리포지토리를 다운로드하려면 `repo_id`와 `repo_type`을 인자로 넘겨주면 됩니다:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp")

snapshot_download(repo_id="google/fleurs", repo_type="dataset")
```

`snapshot_download()` 함수는 기본적으로 최신 버전의 리포지토리를 다운로드합니다. 특정 버전의 리포지토리를 다운로드하고 싶은 경우, `revision` 매개변수에 원하는 버전을 지정하면 됩니다:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### 다운로드할 파일 선택하기

`snapshot_download()` 함수는 리포지토리를 쉽게 다운로드할 수 있도록 해줍니다. 그러나 리포지토리의 모든 내용을 다운로드하고 싶지 않을 수도 있습니다. 예를 들어, `.safetensors` 가중치만 사용하고 싶다면, 모든 `.bin` 파일을 다운로드하지 않도록 할 수 있습니다. `allow_pattern`과 `ignore_pattern` 매개변수를 사용하여 원하는 파일만 다운로드할 수 있습니다.

이 매개변수들은 하나의 패턴이나 패턴의 리스트를 받을 수 있습니다. 패턴은 [여기](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)에서 설명된 것처럼 표준 와일드카드(글로빙 패턴)입니다. 패턴 매칭은 [`fnmatch`](https://docs.python.org/3/library/fnmatch.html)에 기반합니다.

예를 들어, `allow_patterns`를 사용하여 JSON 구성 파일만 다운로드하는 방법은 다음과 같습니다:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

반대로 `ignore_patterns`는 특정 파일을 다운로드에서 제외시킬 수 있습니다. 다음 예제는 `.msgpack`과 `.h5` 파일 확장자를 무시하는 방법입니다:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

마지막으로, 두 가지 매개변수를 함께 사용하여 다운로드를 정확하게 선택할 수 있습니다. 다음은 `vocab.json`을 제외한 모든 json 및 마크다운 파일을 다운로드하는 예제입니다.

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## 로컬 폴더에 파일 다운로드하기

Hub에서 파일을 다운로드하는 가장 좋은 (그리고 기본적인) 방법은 [캐시 시스템](https://huggingface.co/docs/huggingface_hub/ko/guides/manage-cache)을 사용하는 것입니다. 캐시 위치는 `cache_dir` 매개변수로 설정하여 지정할 수 있습니다(`hf_hub_download()`과 `snapshot_download()`에서 모두 사용 가능).

그러나 파일을 다운로드하여 특정 폴더에 넣고 싶은 경우도 있습니다. 이 기능은 `git` 명령어가 제공하는 기능과 비슷한 워크플로우를 만들 수 있습니다. 이 경우 `local_dir`과 `local_dir_use_symlinks` 매개변수를 사용하여 원하는 대로 파일을 넣을 수 있습니다:

- `local_dir`은 시스템 내의 폴더 경로입니다. 다운로드한 파일은 리포지토리에 있는 것과 같은 파일 구조를 유지합니다. 예를 들어 `filename="data/train.csv"`와 `local_dir="path/to/folder"`라면, 반환된 파일 경로는 `"path/to/folder/data/train.csv"`가 됩니다.

- `local_dir_use_symlinks`는 파일을 로컬 폴더에 어떻게 넣을지 정의합니다.
	- 기본 동작(‘자동’)은 작은 파일(5MB 이하)은 복사하고 큰 파일은 심볼릭 링크를 사용하는 것입니다. 심볼릭 링크를 사용하면 대역폭과 디스크 공간을 모두 절약할 수 있습니다. 그러나 심볼릭 링크된 파일을 직접 수정하면 캐시가 손상될 수 있으므로 작은 파일에 대해서는 복사를 사용합니다. 5MB 임계값은 `HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD` 환경 변수로 설정할 수 있습니다.
	- `local_dir_use_symlinks=true`로 설정하면 디스크 공간을 최대한 절약하기 위해 모든 파일이 심볼릭 링크됩니다. 이는 예를 들어 수천 개의 작은 파일로 이루어진 대용량 데이터 세트를 다운로드할 때 유용합니다.
	- 마지막으로 심볼릭 링크를 전혀 사용하지 않으려면 심볼릭 링크를 비활성화하면 됩니다(`local_dir_use_symlinks=False`). 캐시 디렉토리는 파일이 이미 캐시되었는지 여부를 확인하는 데 계속 사용됩니다. 이미 캐시된 경우 파일이 캐시에서 **복사**됩니다(즉, 대역폭은 절약되지만 디스크 공간이 증가합니다). 파일이 아직 캐시되지 않은 경우 파일을 다운로드하여 로컬 디렉터리에 바로 넣습니다. 즉, 나중에 다른 곳에서 다시 사용하려면 **다시 다운로드**해야 합니다.

다음은 다양한 옵션을 요약한 표입니다. 이 표를 참고하여 자신의 사용 사례에 가장 적합한 매개변수를 선택하세요.

|파라미터|캐시되었는지 여부|반환된 파일경로|열람 권한|수정 권한|대역폭의 효율적인 사용|디스크의 효율적인 접근|
|---|:-:|:-:|:-:|:-:|:-:|:-:|
|`local_dir=None`||캐시 속 심볼릭 링크|✅|❌  <br>_(저장하면 캐시가 손상됩니다)_|✅|✅|
|`local_dir="path/to/folder"`  <br>`local_dir_use_symlinks="auto"`||폴더 속 파일 또는 심볼릭 링크|✅|✅ _(소규모 파일의 경우)_  <br>⚠️ _(대규모 파일의 경우 저장하기 전에 경로를 생성하지 마세요)_|✅|✅|
|`local_dir="path/to/folder"`  <br>`local_dir_use_symlinks=True`||폴더 속 심볼릭 링크|✅|⚠️  <br>_(저장하기 전에 경로를 생성하지 마세요)_|✅|✅|
|`local_dir="path/to/folder"`  <br>`local_dir_use_symlinks=False`|아니오|폴더 속 파일|✅|✅|❌  <br>_(다시 실행하면 파일도 다시 다운로드됩니다)_|⚠️  <br>(여러 폴더에서 실행하면 그만큼 복사본이 생깁니다)|
|`local_dir="path/to/folder"`  <br>`local_dir_use_symlinks=False`|예|폴더 속 파일|✅|✅|⚠️  <br>_(파일이 캐시되어 있어야 합니다)_|❌  <br>_(파일이 중복됩니다)_|

