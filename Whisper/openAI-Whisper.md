
## Whisper
---
Whisper는 openAI에서 개발한 **음성 인식 모델**입니다. 이 모델은 다양한 언어를 인식하고 변환할 수 있으며, 특히 음성 텍스트 변환(Speech-To-Text) 작업에 최적화되어 있습니다.

## Setup
---
Whisper를 설치하려면 [[Python]] 과 [[Pytorch]] 가 필요합니다. Whisper는 Python 3.8부터 3.11까지 호환되며, 최신 버전의 Pytorch와도 호환됩니다. 설치는 다음 명령어를 통해 할 수 있습니다.

```bash
pip install -U openai-whisper
```

## Dependency
---
Whisper를 사용하려면 추가로 **ffmpeg** 라는 도구를 설치해야 합니다. ffmpeg는 다양한 운영체제에서 사용할 수 있으며, 설치 방법은 다음과 같습니다.

- ubuntu 또는 Debian

```
sudo apt update && sudo apt install ffmpeg
```

- Arch Linux

```
sudo pacman -S ffmpeg
```

- MacOS

```
brew install ffmpeg
```

- Windows (chocolatey)

```
choco install ffmpeg
```

- Windoes (Scoop)

```
scoop install ffmpeg
```

