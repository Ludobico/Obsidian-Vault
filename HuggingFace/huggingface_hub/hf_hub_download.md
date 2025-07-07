- [[#Hub에서 파일 다운로드하기|Hub에서 파일 다운로드하기]]
	- [[#Hub에서 파일 다운로드하기#파일 하나만 다운로드하기|파일 하나만 다운로드하기]]
	- [[#Hub에서 파일 다운로드하기#최신 버전에서 파일 다운로드하기|최신 버전에서 파일 다운로드하기]]
	- [[#Hub에서 파일 다운로드하기#특정 버전에서 파일 다운로드하기|특정 버전에서 파일 다운로드하기]]
	- [[#Hub에서 파일 다운로드하기#다운로드 URL 만들기|다운로드 URL 만들기]]


## Hub에서 파일 다운로드하기

[[huggingface_hub]] 라이브러리는 Hub의 저장소에서 파일을 다운로드하는 기능을 제공합니다. 이 기능은 함수로 직접 사용할 수 있고, 사용자가 만든 라이브러리에 통합하여 Hub와 쉽게 상호 작용할 수 있도록 할 수 있습니다. 이 가이드에서는 다음 내용을 다룹니다:

- 파일 하나를 다운로드하고 캐시하는 방법
- 리포지토리 전체를 다운로드하고 캐시하는 방법
- 로컬 폴더에 파일을 다운로드하는 방법

### 파일 하나만 다운로드하기

`hf_hub_download()` 함수를 사용하면 Hub에서 파일을 다운로드할 수 있습니다. 이 함수는 원격 파일을 다운로드하여 (버전별로) 디스크에 캐시하고, 로컬 파일 경로를 반환합니다.

### 최신 버전에서 파일 다운로드하기

다운로드할 파일을 선택하기 위해 `repo_id`, `repo_type`, `filename` 매개변수를 사용합니다. `repo_type` 매개변수를 생략하면 파일은 `model` 리포의 일부라고 간주됩니다.

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")

hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
```

### 특정 버전에서 파일 다운로드하기

기본적으로 `main` 브랜치의 최신 버전의 파일이 다운로드됩니다. 그러나 특정 버전의 파일을 다운로드하고 싶을 수도 있습니다. 예를 들어, 특정 브랜치, 태그, 커밋 해시 등에서 파일을 다운로드하고 싶을 수 있습니다. 이 경우 `revision` 매개변수를 사용하여 원하는 버전을 지정할 수 있습니다:

```python
hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**참고**: 커밋 해시를 사용할 때는 7자리의 짧은 커밋 해시가 아니라 전체 길이의 커밋 해시를 사용해야 합니다.

### 다운로드 URL 만들기

리포지토리에서 파일을 다운로드하는 데 사용할 URL을 만들고 싶은 경우 `hf_hub_url()` 함수를 사용하여 URL을 반환받을 수 있습니다. 이 함수는 `hf_hub_download()` 함수가 내부적으로 사용하는 URL을 생성한다는 점을 알아두세요.

