
## torchinfo
---
torchinfo는 [[Pytorch]] 에서 **모델 정보를 시각화하는데 도움을 주는 라이브러리**로, `print(model)` 의 결과를 보완해서 보여줍니다. 이는, TensorFlow의 model.summary() API와 유사한 기능을 제공하여 네트워크를 디버깅할 때 유용합니다.

torchinfo를 사용하면 모델의 계층별 출력 형상, 파라미터 수 및 기타 유용한 정보를 쉽게 확인할 수 있습니다. 이를 통해 모델 구조를 이해하고, 디버깅 과정을 간소화할 수 있습니다. 이 라이브러리는 직관적이고 간단한 인터페이스를 제공하여 프로젝트에 쉽게 통합할 수 있습니다.

## Installation
---

```bash
pip install torchinfo
```

## How to use
---

```python
from torchinfo import summary

model = ConvNet()
batch_size = 16
summary(model, input_size=(batch_size, 1, 28, 28))
```

```
================================================================================================================
Layer (type:depth-idx)          Input Shape          Output Shape         Param #            Mult-Adds
================================================================================================================
SingleInputNet                  [7, 1, 28, 28]       [7, 10]              --                 --
├─Conv2d: 1-1                   [7, 1, 28, 28]       [7, 10, 24, 24]      260                1,048,320
├─Conv2d: 1-2                   [7, 10, 12, 12]      [7, 20, 8, 8]        5,020              2,248,960
├─Dropout2d: 1-3                [7, 20, 8, 8]        [7, 20, 8, 8]        --                 --
├─Linear: 1-4                   [7, 320]             [7, 50]              16,050             112,350
├─Linear: 1-5                   [7, 50]              [7, 10]              510                3,570
================================================================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
Total mult-adds (M): 3.41
================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 0.40
Params size (MB): 0.09
Estimated Total Size (MB): 0.51
================================================================================================================
```

## Parameters
---

```python
def summary(
    model: nn.Module,
    input_size: Optional[INPUT_SIZE_TYPE] = None,
    input_data: Optional[INPUT_DATA_TYPE] = None,
    batch_dim: Optional[int] = None,
    cache_forward_pass: Optional[bool] = None,
    col_names: Optional[Iterable[str]] = None,
    col_width: int = 25,
    depth: int = 3,
    device: Optional[torch.device] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    mode: str | None = None,
    row_settings: Optional[Iterable[str]] = None,
    verbose: int = 1,
    **kwargs: Any,
) -> ModelStatistics:
```

> model -> nn.Module
- 요약할 Pytorch 모델입니다. 모델은 반드시 `train()` 또는 `eval()` 모드 중 하나여야 합니다.

> input_size
- 입력 데이터의 형태를 List/Tuple/[[torch.size]] 로 지정합니다. 배치 크기도 포함해야합니다.

> input_data
- 모델의 `forward()` 함수에 대한 인수들입니다. 여러 매개변수가 필요한 경우 리스트나 딕셔너리 형태로 전달할 수 있습니다.

> batch_dim
- 입력 데이터의 배치 차원을 지정합니다. 지정하지 않으면 input_data 또는 input_size에 포함된 배치차원을 사용합니다.

> cache_forward_pass
- `True` 로 설정하면 `forward()` 함수의 실행 결과를 캐시합니다. Jupyter 노트북에서 모델 요약의 형식을 변경할 때 유용합니다.

> col_names
- 출력에 표시할 열을 지정합니다.
- "input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable" 을 지원합니다.

> col_width
- 각 열의 너비를 설정합니다.

> depth
- 표시할 중첩 레이어의 깊이를 설정합니다. 이 depth 아래의 중첩 레이어는 요약에 표시되지않습니다.

> device
- 모델과 input_data에 사용할 torch device를 지정합니다. 지정하지 않으면 input_data 또는 모델 파라미터의 dtype을 사용합니다.

> dtypes
- input_size를 사용할 때 torchinfo는 기본적으로 FloatTensors를 사용합니다. 다른 데이터 타입을 사용하는 경우 dtype을 지정합니다.

> mode
- "train" 또는 "eval" 모드 중 하나를 지정합니다. `summary()` 를 호출하기전에 model.train() 또는 model.eval()을 호출합니다.

> row_settings
- 행에 표시할 기능을 지정합니다.
- "ascii_only" , "depth", "var_names" 를 지원합니다.

> verbose
- 출력 수준을 설정합니다.
	- 0 : 출력 없음
	- 1 : 모델 요약 출력
	- 2 : 가중치와 바이어스 레이어를 자세히 출력

> kwargs
- `model.forward` 함수에 사용되는 기타 인수입니다.

