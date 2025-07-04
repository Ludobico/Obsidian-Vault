
`XttsConfig` 는 [[TTS&STT/xtts/xtts]] 의 [[TTS&STT/xtts/TTS/TTS|TTS]] 모델 config를 정의하는데 사용되는 클래스입니다. 이 클래스는 **text to speech 과정에서 다양한 파라미터를 조정** 할 수 있도록 설계되었습니다. 아래는 주어진 파라미터들에 대한 설명입니다.

## Parameters

> model -> str

모델의 이름을 지정합니다.
모델 이름은 변경하면 예상치 못한 동작이 발생할 수 있으므로, 확실한 이유가 없으면 xtts에서 제공하는 기본값을 유지하는 것이 좋습니다.

> model_args -> [[XttsArgs]]

모델의 아키텍처와 관련된 추가 설정입니다.
기본값은 [[XttsArgs]] 로 사전에 정의된 기본 아키텍처 설정을 사용합니다.

> audio -> XttsAudioConfig

오디오 처리와 관련된 설정을 정의합니다.
기본값은 `XttsAudioConfig()` 로, 표준 오디오 config를 사용합니다.

> model_dir -> str, default : None

XTTS 모델 파일들이 위치한 디렉토리 경로입니다.
기본값은 `None` 으로 지정되어 있으며, 경로를 명시해야 모델이 정상 작동합니다.

> temperature -> float, default : 0.2

[[temperature]] 를 정의합니다. 높은 값은 더 창의적인 결과를 생성하지만 stability가 감소합니다.

> length_penalty -> float

생성된 텍스트의 길이에 패널티를 적용하여 길이 조정을 제어합니다.
값이 `> 0` 일 경우 긴 텍스트를 선호하며, `< 0` 일 경우 짧은 텍스트를 선호합니다.

> repetition_penalty -> float, default : 2.0

반복되는 패턴을 억제하기 위한 패널티를 설정합니다.
값이 `1.0` 이면 패널티가 없습니다.

> top_p -> float, default : 0.8

생성 과정에서 가장 높은 확률을 가진 토큰 집합을 유지하는 값인 [[top_p]] 를 설정합니다.

> num_gpt_outputs -> int, default : 16

GPT 모델에서 샘플링된 outputs의 갯수입니다.
outputs의 수가 많을수록 더 창의적이고 다양한 결과를 얻을 가능성이 높아집니다.

> gpt_cond_len -> int, default : 12

AutoRegressive 모델에 조건을 부여하기 위해 사용하는 **오디오 길이(초)** 입니다.

> gpt_cond_chunk_len -> int, default : 4

오디오를 청크 단위로 나누는 크기입니다. 값은 `<= gpt_cond_len` 이어야하며, 청크를 나누면 안정성이 증가합니다.

> max_ref_len -> int, default : 10

디코더에 사용할 수 있는 **최대 오디오 길이(초)** 입니다.

> sound_norm_refs -> bool, default : False

조건 오디오를 정규화할지 여부를 결정합니다.

```python
from TTS.tts.configs.xtts_config import XttsConfig
config = XttsConfig()
```

