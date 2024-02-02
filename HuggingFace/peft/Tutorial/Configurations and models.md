[[peft]] (Parameter-Efficient Fine-Tuning) 라이브러리는 오늘날의 대규모 pre-trained 모델의 크기 때문에 발생하는 문제에 대응하기 위해 설계되었습니다. 이러한 모델은 보통 수십억~수백억개의 파라미터를 가지며, 이러한 모델을 train 하기 위해서는 많은 저장 공간과 계산 능력이 필요합니다. 이는 강력한 GPU 또는 TPU 에 액세스해야하며, 비용이 많이 들고 실용적이지가 않습니다. PEFT는 이러한 많은 문제를 해결할 수 있는 방법을 제공합니다.

PEFT 에는 <font color="#00b050">소프트 프롬프팅(soft prompting)</font>, <font color="#00b050">행렬 분해(Matrix decomposition)</font>, <font color="#00b050">어댑터(adapters) </font>등 여러 유형이 있지만, 모두 <font color="#ffff00">매개변수의 수를 줄이는 것에 중점</font>을 둡니다.

PEFT 라이브러리는 무료 GPU에서 빠르게 대규모 모델을 훈련할 수 있도록 설계되었습니다. 이 튜토리얼에서는 PEFT 방법을 pre-trained model에 적용하여 train 하기 위한 config를 설정하는 방법을 배우게 됩니다. PEFT config가 설계되면 [[Trainer]] 클래스 및 [[Pytorch]] train loop 와 같은 원하는 train framework를 사용할 수 있습니다.

## PEFT Configurations
---
PEFT의 configuration은 특정 PEFT 방법이 적용되는 방식을 지정하는 중요한 매개변수를 저장하는 역할을 합니다. 각 PEFT 방법에 대한 구성을 조정할 수 있으며, 이를 통해 해당 방법을 최적화하거나 특정 작업게 맞게 조정할 수 있습니다. PEFT 구성은 JSON 형시그로 제공되며, 이를 통해 사용자는 각 PEFT 방법의 동작을 세부적으로 제어할 수 있습니다.

예를 들어, [[LoRA]] 방법을 적용하는 경우 LoraConfig, p-tuning 을 적용하는 경우에는 PromptEncoderConfig 와 같은 config 파일을 사용할 수 있습니다. 이러한 config 파일에는 해당 방법에 필요한 매개변수들이 포함되어 있습니다.

### LoraConfig

```json
{
  "base_model_name_or_path": "facebook/opt-350m", #base model to apply LoRA to
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA", #PEFT method type
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj", #model modules to apply LoRA to (query and value projection layers)
    "v_proj"
  ],
  "task_type": "CAUSAL_LM" #type of task to train model on
}
```

### PromptEncoderConfig

```json
{
  "base_model_name_or_path": "roberta-large", #base model to apply p-tuning to
  "encoder_dropout": 0.0,
  "encoder_hidden_size": 128,
  "encoder_num_layers": 2,
  "encoder_reparameterization_type": "MLP",
  "inference_mode": true,
  "num_attention_heads": 16,
  "num_layers": 24,
  "num_transformer_submodules": 1,
  "num_virtual_tokens": 20,
  "peft_type": "P_TUNING", #PEFT method type
  "task_type": "SEQ_CLS", #type of task to train model on
  "token_dim": 1024
}
```


