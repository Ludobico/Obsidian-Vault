
`from_config` 메서드는 [[HuggingFace🤗]] 의 [[transformers]] 라이브러리에서 특정 구성([[PretrainedConfig]]) 에 기반하여 모델 인스턴스를 생성하는 데 사용됩니다. 이 메서드는 주로 특정 설정을 가진 모델을 처음부터 초기화할 때 사용되며, 사전 훈련된 가중치를 로드하는 대신 사용자가 제공한 구성을 기반으로 모델을 설정합니다.

> config -> [[PretrainedConfig]] 

- 모델을 인스턴스화할 때 사용될 config 객체입니다. 이 객체는 특정 모델 클래스의 설정을 정의하고, 해당 설정에 맞는 모델 클래스를 자동으로 선택합니다.

```python
from transformers import AutoConfig, AutoModelForCausalLM

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForCausalLM.from_config(config)

```

