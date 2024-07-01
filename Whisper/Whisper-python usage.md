
## load Whipser model
---

```python
import whisper

model = whisper.load_model("base")
```

`whipser.load_model()` 은 [[openAI-Whisper]] 모델을 로드할 수 있습니다.

현재 로드할 수 있는 모델은 다음과 같습니다.

| Size   | Parameters | English-Only model | Multilingual model | Required VRAM | Relative speed |
| ------ | ---------- | ------------------ | ------------------ | ------------- | -------------- |
| tiny   | 39 M       | tiny.en            | tiny               | ~ 1 GB        | ~32x           |
| base   | 74 M       | base.en            | base               | ~ 1 GB        | ~16x           |
| small  | 244 M      | small.en           | small              | ~ 2 GB        | ~6x            |
| medium | 769 M      | medium.en          | medium             | ~ 5 GB        | ~2x            |
| large  | 1550 M     |                    | large              | ~ 10 GB       | 1x             |

## Speech to text convert
---

```python
result = model.transcribe("audio.mp3")
print(result["text"])
```

`model.transcribe("audio.mp3")` 메서드는 음성 파일을 텍스트로 변환합니다. 반환된 `result` 딕셔너리에 변환된 텍스트는 `result['text']` 로 접근할 수 있습니다.

