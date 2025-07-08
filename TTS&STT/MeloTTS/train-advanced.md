이 문서는 [[MeloTTS]] 학습 이전에 수행된 전처리 과정, 관련 코드, 그리고 학습 준비 전 단계의 전반적인 내용을 정리한 것입니다.

## Precondition

- [[MeloTTS]]
- mecab (리눅스 기반인 경우)

mecab 설치는 아래의 커맨드로 설치를 진행합니다.

```bash
apt-get update && apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
```

```
pip install python-mecab-ko
```


## Step 1 : prepare files

옵시디언의 [[MeloTTS]] 문서에 첨부된 `train_preprocess.py`와 `dataset_preprocess.py`, `getenv.py` 파일을 MeloTTS 루트 디렉토리로 복사합니다.

![[Pasted image 20250708114810.png]]

### dataset_preprocess.py

```python
env = GetEnv()
```

전체적으로 사용할 데이터셋을 저장할 폴더 경로를 지정합니다. 예시 코드처럼 별도로 지정되지 않으면, `getenv.py` 파일이 위치한 디렉토리를 기준으로 폴더가 생성됩니다.

폴더는 다음과 같은 형태로 자동 생성됩니다.

```
train
	dataset
	models
	output
```

그 후, [[HuggingFace🤗]]에서 음성-텍스트 기반 데이터셋을 검색한 뒤, 원하는 데이터셋의 `repo_id`를 아래 `main.py` 코드에 추가합니다:

```python
if __name__ == "__main__":
    repo_id = "habapchan/genshin-nahida-korean"
    make_meloTTS_dataset(repo_id)
```

`make_meloTTS_dataset` 함수는 HuggingFace에서 불러온 데이터셋을 MeloTTS 형식에 맞게 전처리하는 함수이며, 다음과 같은 `kwargs`를 인자로 받을 수 있습니다.

> text_key -> str , default : transcription

- 텍스트 정제를 적용할 Key 값의 이름입니다. HuggngFace의 데이터셋 형태에서 확인할 수 있습니다.

![[Pasted image 20250708120512.png]]

> speaker_name -> str, default : "KR-default"

- [[MeloTTS]] 학습에 사용되는 메타데이터입니다. [[train]] 에서 확인할 수 있습니다.

> language_code -> str, default : "KR"

- [[MeloTTS]] 학습에 사용되는 메타데이터입니다. [[train]] 에서 확인할 수 있습니다.

> apply_re -> bool, default : True

- 텍스트를 정제(regex cleaning)을 할지 결정하는 불리언 값입니다. `True` 일 경우 `,` `.` 을 제외한 특수문자가 제거됩니다.

> make_wav -> bool, default : True

- 데이터셋의 오디오를 `.wav` 파일로 저장할지 여부를 결정합니다. wav 파일의 이름은 `korean_{number}.wav` 로 저장됩니다.

> verbose -> bool, default : False

- 데이터셋의 정보를 출력할지 여부를 결정합니다.

