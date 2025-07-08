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


## Step 1

옵시디언의 [[MeloTTS]] 내에 동봉된 `train_preprocess.py` 파일과 `dataset_preprocess.py` 파일을 MeloTTS 루트프로젝트로 복사합니다.

