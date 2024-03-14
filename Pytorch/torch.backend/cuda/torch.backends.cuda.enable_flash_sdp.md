**torch.backends.cuda.enable_flash_sdp()** 메서드는 flash scaled dot product attention을 활성화 또는 비활성화하는 기능을 제공합니다.

flash scaled dot product attention은 어텐션 매커니즘 중 하나로, 주로 [[Transformer]] 와 같은 모델에서 사용됩니다.

사용 예시는 다음과 같습니다.

```python
import torch

# 플래시 스케일드 닷 프로덕트 어텐션 활성화
torch.backends.cuda.enable_flash_sdp(True)

# 플래시 스케일드 닷 프로덕트 어텐션 비활성화
torch.backends.cuda.enable_flash_sdp(False)
```

### no kernel found to launch

위와 같은 에러코드에서 아래의 코드처럼 두 가지 cuda 옵션을 비활성화 하여 해결할 수 있습니다.

```python
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
```