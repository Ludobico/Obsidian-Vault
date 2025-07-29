
## Training

학습을 시작하기 전에, [[MeloTTS]] 를 설치한 뒤 `melo` 폴더로 이동하세요.

```python
pip install -e .
python -m unidic download
cd melo
```

## Data Preparation

TTS 모델을 학습하려면 오디오 파일들과 메타데이터 파일을 준비해야 합니다.
오디오는 **44100Hz** 형식을 궈장하며, 메타데이터 파일은 다음과 같은 형식을 가져야 합니다.

```
path/to/audio_001.wav |<speaker_name>|<language_code>|<text_001>
path/to/audio_002.wav |<speaker_name>|<language_code>|<text_002>
```

여기서 `<text>` 는 음성 인식 모델(예 : [[Whisper]])를 사용하여 텍스트로 변환할 수 있습니다.
예시 메타데이터는 `data/example/metadata.list` 에 있습니다.

그다음 아래 명령어로 전처리를 실행할 수 있습니다.

```bash
python preprocess_text.py --metadata data/example/metadata.list 
```

이 작업을 통해 `data/example/config.json` config 파일이 생성됩니다.
해당 파일의 **하이퍼파라미터를 자유롭게 수정**할 수 있으며, 예를 들어 CUDA 메모리 부족 에러가 발생할 경우 배치 사이즈를 줄이는 것이 좋습니다.

## Train

학습은 다음 명령어로 시작할 수 있습니다.

```bash
bash train.sh <path/to/config.json> <num_of_gpus>
```

일부 머신에서는 gloo 관련 문제로 학습이 중단될 수 있습니다.
이를 방지하기 위해 `train.sh` 에 자동 재시작 래퍼(wrapper)가 추가되어 있습니다.

## Inference

아래 명령어로 추론을 수행할 수 있습니다.

```bash
python infer.py --text "<some text here>" -m /path/to/checkpoint/G_<iter>.pth -o <output_dir>
```

## Tips

melotts에서 그냥 학습 돌리면 자동으로 설치되는 모델이 있는데 그걸로는 잘 안됩니다

MeloTTS안에 melo폴더에 download_utils.py 열어보면

다운로드 DOWNLOAD_CKPT_URL가 있는데

CKPT받으시면 여기서 KR에 있는거 받으시고 그걸 G_0.pth로 이름을 바꾸시고

밑에 PRETRAINED_MODELS에서 D.pth랑 DUR.pth받으신 후에

학습폴더 안에 넣으시면 한국어 학습이 됩니다

