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
pip install matplotlib==3.7.0
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

- [[MeloTTS]] 학습에 사용되는 메타데이터입니다. [[train_from_github]] 에서 확인할 수 있습니다.

> language_code -> str, default : "KR"

- [[MeloTTS]] 학습에 사용되는 메타데이터입니다. [[train_from_github]] 에서 확인할 수 있습니다.

> apply_re -> bool, default : True

- 텍스트를 정제(regex cleaning)을 할지 결정하는 불리언 값입니다. `True` 일 경우 `,` `.` 을 제외한 특수문자가 제거됩니다.

> make_wav -> bool, default : True

- 데이터셋의 오디오를 `.wav` 파일로 저장할지 여부를 결정합니다. wav 파일의 이름은 `korean_{number}.wav` 로 저장됩니다.

> verbose -> bool, default : False

- 데이터셋의 정보를 출력할지 여부를 결정합니다.


#### result

`__main__` 구문의 커맨드를 실행시키면, 아래와 같은 결과가 출력됩니다.

```
DatasetDict({
    train: Dataset({
        features: ['audio', 'speaker', 'language', 'transcription'],
        num_rows: 1476
    })
})
1476it [01:32, 15.95it/s]
Please run the following command manually:
cd path\MeloTTS\melo 
python preprocess_text.py --metadata path\MeloTTS\train\dataset\genshin-nahida-korean\metadata.list
```

```
train
	dataset
		repo_without_id
			korean_0.wav
			korean_1.wav
			...
			metadata.list
		repo_id
	models
	output
```

커맨드에 출력된 아래 명령어를 순서대로 입력합니다.

```
cd path\MeloTTS\melo 
python preprocess_text.py --metadata path\MeloTTS\train\dataset\genshin-nahida-korean\metadata.list
```

`train/dataset` 디렉토리에 아래와 같은 추가파일이 생성되었는지 확인합니다.

```
config.json
train.list
val.list
각 korea_{number} 에 대응하는 pt 파일
```

`config.json` 에서 학습할 파라미터를 지정합니다.

```json
{
  "train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 52,
    "epochs": 10000,
    "learning_rate": 0.0003,
    "betas": [
      0.8,
      0.99
    ],
    "eps": 1e-09,
    "batch_size": 6,
    "fp16_run": false,
    "lr_decay": 0.999875,
    "segment_size": 16384,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "skip_optimizer": true
  },
  ...
  
```

### train_preprocess.py

`if __name__ == "__main__"` 구문에서 `train/dataset/repo_without_id` 를 `target_dir` 로지정한 뒤, 실행합니다.

```python
if __name__ == "__main__":
    target_dir = r"path\genshin-nahida-korean"
    prepare_pretrained_models(target_dir)
```

```
  ✓ config.json exists
  ✓ D.pth exists
  ✓ DUR.pth exists
  ✓ G_0.pth exists
  ✓ metadata.list exists
  ✓ metadata.list.cleaned exists
  ✓ train.list exists
  ✓ val.list exists
Please run the following command manually:
cd path\MeloTTS\melo 
bash train.sh path\MeloTTS\train\dataset\genshin-nahida-korean/config.json <num_of_gpus>
```


## Important

맨 아래 커맨드

```
bash train.sh path\MeloTTS\train\dataset\genshin-nahida-korean/config.json <num_of_gpus>
```

를 실행시키면 meloTTS/melo에  **logs** 라는 폴더가 생성되면서 추가적으로

학습이 시작되면 **D_0.pth**, **DUR_0.pth**, **G_0.pth** 파일이 생성됩니다.  
세 파일이 모두 생성되고 첫 번째 epoch가 시작되면 학습을 중단한 뒤, 기존에 `prepare_pretrained_models.py`를 통해 다운로드한 **G_0.pth** 파일로 `logs` 디렉토리에 있는 해당 파일을 덮어씌웁니다.


![[모델 복사.png|512]]

![[모델 붙여넣기.png]]

덮어씌우는게 완료되었다면, `train.log` 파일을 삭제하고 다시한 번

```
bash train.sh path\MeloTTS\train\dataset\genshin-nahida-korean/config.json <num_of_gpus>
```

아래 커맨드를 통해 학습을 진행합니다.

## train이 시작되도 모델파일이 보이지 않는다면 또는 모델파일을 직접 지정하고 싶다면

- `train.sh` 스크립트가 이제 `--pretrain_G`, `--pretrain_D`, `--pretrain_dur`까지 받도록 수정됨.
    
- 그런데 이 인자들이 _빈 문자열일 경우_, `torchrun` 명령어에서는 **해당 옵션들을 생략하고 싶음**.
    
- 따라서 `train.sh`를 실행할 때는 **선택적 인자**를 허용하는 방식이 필요함.

```bash
CONFIG=$1
GPUS=$2
PRETRAIN_G=$3
PRETRAIN_D=$4
PRETRAIN_DUR=$5

MODEL_NAME=$(basename "$(dirname "$CONFIG")")
PORT=10902

# 인자 조건부 처리
PRETRAIN_ARGS=""
[ -n "$PRETRAIN_G" ] && PRETRAIN_ARGS="$PRETRAIN_ARGS --pretrain_G $PRETRAIN_G"
[ -n "$PRETRAIN_D" ] && PRETRAIN_ARGS="$PRETRAIN_ARGS --pretrain_D $PRETRAIN_D"
[ -n "$PRETRAIN_DUR" ] && PRETRAIN_ARGS="$PRETRAIN_ARGS --pretrain_dur $PRETRAIN_DUR"

while :
do
  torchrun --nproc_per_node=$GPUS \
           --master_port=$PORT \
           train.py --config "$CONFIG" \
                    --model "$MODEL_NAME" \
                    $PRETRAIN_ARGS

  for PID in $(ps -aux | grep "$CONFIG" | grep python | awk '{print $2}')
  do
    echo $PID
    kill -9 $PID
  done
  sleep 30
done
```

### 사용법

**기존처럼 pretrain 없이 실행하려면:**


```bash
bash train.sh path/to/config.json 2
```


- `$3`, `$4`, `$5`가 비어 있기 때문에 `--pretrain_*` 인자들은 자동으로 생략됩니다.
    

**pretrain 모델 경로를 지정하려면:**

```bash
bash train.sh path/to/config.json 2 path/to/G.pth path/to/D.pth path/to/DUR.pth
```

- 경로가 비어 있지 않으므로 자동으로 인자들이 붙게 됩니다.

