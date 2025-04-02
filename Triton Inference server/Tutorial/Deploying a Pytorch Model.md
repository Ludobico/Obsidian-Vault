여기에서는 간단한 ResNet 모델을 [[Triton Inference server]] 에 배포하는 방법을 설명합니다.

## Step 0 : Clone Repository

먼저 [tutorial repository](https://github.com/triton-inference-server/tutorials) 를 클론합니다.

```
git clone https://github.com/triton-inference-server/tutorials.git
```

## Step 1 : Export the model

${PWD} 에서는 tutorial/Quick_Deploy/PyTorch의 경로를 입력합니다.

```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's PyTorch
# container; eg. 22.04

docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:<xx.xx>-py3
python export.py
```

## Step 2 : Set up Triton Inference Server

모델 레파지토리 설정은 Triton을 사용하기 위한 핵심적인 첫 단계입니다.

### 모델 레파지토리 구조

```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.pt
```

- model_repository
	- 이 폴더는 **triton이 모델을 인식하고 관리하는 최상위 디렉토리**입니다. 여러 모델을 한꺼번에 배포하고 싶을 때, 각 모델별로 하위 폴더를 만들어 넣을 수 있습니다.
- resnet50
	- 이건 특정 모델의 이름입니다. 예제에서는 ResNet-50이라는 이미지 분류 모델을 사용합니. 다른 모델을 배포하고 싶다면 이 **폴더 이름을 해당 모델 이름**으로 바꾸시면 됩니다.
- config.pbtxt
	- 모델의 설정을 정의하는 config 파일입니다. Triton이 모델을 실행하려면 입력과 출력의 형태, 사용하는 프레임워크 (예 : [[Pytorch]]), 데이터 타입 같은 정보가 필요합니다. 이 설정을 정확히 작성해야 서버가 제대로 동작합니다.
	- 공식문서의 [review Part 1](https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_1-model_deployment/README.md) 을 참고하면 이 파일을 어떻게 구성하는지 예제와 함께 자세히 나와 있습니다.
- 1
	- 모델 버전을 나타내는 폴더입니다. Triton은 버전 관리를 지원해서, 모델을 업데이트할 때마 다 `1`, `2`, `3` 처럼 버전별로 폴더를 나눠 관리할 수 있습니다. 현재는 `1` 이라는 첫 번째 버전만 있는 상태입니다.
- model.pt
	- 실제 모델 파일입니다. 여기에서는 pytorch로 저장된 `.pt` 파일을 예로 들었지만, 사용하는 프레임워크에 따라 `.onnx` , `.pb` 같은 다른 형식의 파일이 들어갈 수도 있습니다.

```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## Step 3 : Using a Triton Client to Query the server

다른 커맨드창을 열어서 실행합니다.

### 의존성 설치 및 테스트 이미지 다운로드

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk bash

pip install torchvision

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

- triton의 Python SDK 이미지를 실행합니다.
- `--net=host` 는 컨테이너가 호스트의 네트워크를 그대로 사용하게 합니다. 포트 매핑 없이 `localhost` 로 바로 통신할 수 있습니다.
- `bash` 컨테이너가 시작되자마자 Bash 셸을 실행합니다.

레파지토리안에 있는 client.py를 실행합니다.

```
python client.py
```

결과값은 이렇게 나올것입니다.

```
[b'12.468750:90' b'11.523438:92' b'9.664062:14' b'8.429688:136'
 b'8.234375:11']
```

