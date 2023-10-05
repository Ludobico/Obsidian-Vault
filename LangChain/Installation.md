python 을 기준으로 `langchain` 을 설치하려면 아래의 명령어를 입력하세요
```
pip install langchain 
# or
conda install langchain -c conda-forge
```

# Installation With LlamaCPP
LlamaCPP 란 C/C++ 로 작성된 Llama 모델을 실행시키는 프로그램으로 interger quantization 또는 float quantization 으로 모델 아키텍처를 구성할 수 있습니다. quantization 을 통해 만들어진 모델을 ggml 이라고 하는데, ggml의 목적은 CPU로도 연산이 가능한 모델을 개발하는것이기 때문에 langchain 과 LlamaCPP 를 통해 CPU를 기반으로 언어 모델 어플리케이션을 개발할 수 있습니다.

⚠️ 최소 16GB 이상의 메모리 용량이 필요합니다.

아래의 명령어를 통해 `LlamaCPP` 를 설치할 수 있습니다.
```
pip install llama-cpp-python
```

