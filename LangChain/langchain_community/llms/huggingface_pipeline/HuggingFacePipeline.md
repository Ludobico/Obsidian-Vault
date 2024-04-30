`HuggingFacePipeline` 은 [[HuggingFace🤗]] 에서 제공하는 모델을 [[LangChain/LangChain|LangChain]] 에서 사용할 수 있게 해주는 클래스입니다. 이를 통해 사용자는 Huggingface hub에 호스팅된 수많은 모델을 직접 다운로드하고 실행할 수 있습니다.

`HuggingFacePipeline` 을 사용하려면 [[transformers]] 및 [[Pytorch]] 패키지를 설치해야 합니다. 메모리 효율성을 높이기 위해 xformer 패키지를 추가로 설치할 수 있습니다.

## example code
---

```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)
```

## example code 2
---

```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)
```

