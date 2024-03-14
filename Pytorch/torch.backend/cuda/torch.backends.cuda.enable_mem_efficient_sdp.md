**torch.backends.cuda.enable_mem_efficient_sdp()** 메서드는 memory efficient scaled dot product ateention 을 활성화 또는 비활성화하는 기능을 제공합니다.

scaled dot product attention은 주로 [[Transformer]] 와 같은 모델에서 사용되는 어텐션 매커니즘 입니다. 이 어텐션은 입력 벡터 간의 상대적인 중요성을 계산하는 데 사용됩니다.

사용 예시는 다음과 같습니다.

```python
import torch

# 메모리 효율적인 스케일드 닷 프로덕트 어텐션 활성화
torch.backends.cuda.enable_mem_efficient_sdp(True)

# 메모리 효율적인 스케일드 닷 프로덕트 어텐션 비활성화
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

### no kernel found to launch

위와 같은 에러코드에서 아래의 코드처럼 두 가지 cuda 옵션을 비활성화 하여 해결할 수 있습니다.

```python
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
```

