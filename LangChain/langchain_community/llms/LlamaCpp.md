- [[#Llamap.cpp|Llamap.cpp]]
- [[#Installation|Installation]]
	- [[#Installation#CPU-Only|CPU-Only]]
	- [[#Installation#GPU (cuBLAS, OpenBLAS)|GPU (cuBLAS, OpenBLAS)]]
- [[#Installation on Windows|Installation on Windows]]
- [[#Error handle|Error handle]]
	- [[#Error handle#TypeError: 'NoneType' object is not callable|TypeError: 'NoneType' object is not callable]]
		- [[#TypeError: 'NoneType' object is not callable#langchain|langchain]]
		- [[#TypeError: 'NoneType' object is not callable#llama-cpp-python|llama-cpp-python]]

## Llamap.cpp 

`llama-cpp-python` 은 `llama.cpp` 를 [[Python]] 에서 사용할 수 있도록 해주는 바인딩 라이브러리입니다. [[HuggingFace🤗]] 에 올라와있는 다양한 LLM 에 대한 추론을 지원하며, [[LangChain/LangChain|LangChain]] 과 통합하여 사용할 수 있습니다.

참고로 `llama-cpp-python` 의 최신 버전은 기존 GGML 포맷대신 [[GGUF]] 포맷 파일 형식을 사용합니다.

만일 **기존 GGML 모델을 GGUF 형식으로 변환**하려면, `llama.cpp` 에서 다음 명령어를 실행합니다.

```
python ./convert-llama-ggmlv3-to-gguf.py --eps 1e-5 --input models/openorca-platypus2-13b.ggmlv3.q4_0.bin --output models/openorca-platypus2-13b.gguf.q4_0.bin
```

## Installation

`llma-cpp-python` 은 다양한 방식으로 설치할 수 있으며, 사용하는 환경(CPU, GPU, Mac) 에 따라 선택할 수 있습니다.

### CPU-Only

```
pip install --upgrade --quiet llama-cpp-python
```


### GPU (cuBLAS, OpenBLAS)

`llma.cpp` 는 성능 향상을 위해 여러 BLAS 백엔드를 지원합니다.
아래와 같이 환경변수를 설정하여 GPU 지원을 활성화한 상태로 설치할 수 있습니다.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

<font color="#ffff00">이전에 CPU-Only 버전을 설치했다면, 반드시 아래처럼 강제 재설치해야 합니다.</font>

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

## Installation on Windows

Windows에서는 소스 코드에서 직접 빌드해서 설치하는 방식이 가장 안정적입니다.

필수 준비
> 	git
	python
	cmake
	Visual Studio Community (make sure you install this with the following settings)
		Desktop development with C++
		Python development
		Linux embedded development with C++

리포지토리 복제 및 설정

```bash
git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
```

환경 변수 설정

```
set FORCE_CMAKE=1  
set CMAKE_ARGS=-DGGML_CUDA=ON
```

설치 명령

```bash
python -m pip install -e .
```

<font color="#ffff00">이전에 CPU-Only 버전을 설치했다면, 반드시 아래처럼 강제 재설치해야 합니다.</font>

```bash
python -m pip install -e . --force-reinstall --no-cache-dir
```


## Error handle

### TypeError: 'NoneType' object is not callable

[관련 깃허브 이슈](https://github.com/abetlen/llama-cpp-python/issues/1610)

[[Python]] 인터프리터가 종료될 때, 객체와 모듈의 삭제 순서는 보장되지 않습니다. 따라서 `_LlamaModel.__del__()` 이 실행될 때 `llama_cpp` 모듈이 이미 사라졌다면 `llma_cpp.llama_free.model()` 호출이 실패하게 됩니다.

이를 해결하려면 `__del__()` 에 의존하지 않고, <font color="#ffff00">atexit</font> 모듈을 사용해 명시적으로 `model.close()` 를 호출해야합니다.

#### langchain

```python
import atexit
from langchain_community.llms import LlamaCpp

model = LlamaCpp(...)

@atexit.register
def free_model():
    model.client.close()
```

#### llama-cpp-python

```python
import atexit
from llama_cpp import Llama

model = Llama(...)

@atexit.register
def free_model():
    model.close()
```
