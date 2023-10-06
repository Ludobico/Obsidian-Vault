이 문서는 [[Detrex]] 프레임워크를 사용자가 이용할 수 있도록 설치할 수 있는 가이드라인을 제시합니다.

## <font color="#ffc000">Requirements</font>

- <font color="#ffff00"> Python 3.7</font> 버전 이상
- <font color="#ffff00"> Pytorch 1.10 버전 이상</font> CUDA 버전이 12.1 버전이면 Detectron2의 [[Detectron2/Installation|Installation]] 가이드를 참조하세요

## <font color="#ffc000">Build detrex from source</font>

Detrex를 사용하기 위해선 Detectron2가 SubModule로 필요합니다. 아래의 설치과정은 아나콘다 가상환경을 예시로 설치합니다.

먼저, 아나콘다 가상환경을 생성합니다.
```bash
conda create -n detrex python=3.7 -y
conda activate detrex
```

detrex 깃 레파지토리를 복사하고, detectron2 프레임워크를 서브모듈로 초기화합니다.
```bash
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init
git submodule update
```

detrex 및 detectron2의 의존성과 서브모듈 패키지를 업데이트합니다.
```bash
python -m pip install -e detectron2
pip install -e .
```

## <font color="#ffc000">Verity the installation</font>

detrex가 올바르게 설치되었는지 확인하기 위해, 예시로 샘플 데이터, 모델, 코드를 제공하겠습니다.
아래의 코드에서는 <font color="#ffff00">wget</font> 명령어를 사용합니다.

- <font color="#ffc000">step 1</font>. 샘플이미지와 사전훈련된 모델을 다운로드받습니다.
```bash
cd detrex

# download pretrained DAB-DETR model
wget https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_r50_50ep.pth

# download pretrained DINO model
wget https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_r50_4scale_12ep.pth

# download the demo image
wget https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/idea.jpg
```

- <font color="#ffc000">step 2</font>. detrex에 설치된 <font color="#ffff00">demo.py</font> 파이썬파일로 데이터를 평가합니다.
```bash
python demo/demo.py --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py \
                    --input "./idea.jpg" \
                    --output "./demo_output.jpg" \
                    --opts train.init_checkpoint="./dab_detr_r50_50ep.pth"
```

![[demo_output.jpg]]