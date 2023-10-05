[[Detectron2]]를 설치하기에 앞서 아래의 요구사항이 필요합니다.

- <font color="#00b050">python 3.7 이상의 버전</font>
- [[pytorch]] <font color="#00b050">1.8 이상의 버전</font>
- [[OpenCV]] 

CUDA 버전 및 pytorch 버전에 맞게 아래의 터미널 명령어로 설치할 수 있습니다.

> [Linux only]
- CUDA 11.3 , torch 1.10
```bash
python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html"
```

- CUDA 11.1 , torch 1.10
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

- CUDA 11.1 , torch 1.9
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

- CUDA 11.1 , torch 1.8
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```


> [Windows]
```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


> CUDA 12.1 version

pytorch cu118의 경우 cuda 11.9, 12.0, 12.1 모두 지원이 되지만 [[Detectron2]] 는 단순히 CUDA 버전과 pyrorch cu118의 <font color="#ffff00">숫자만 비교</font> 하여 일치하지 않으며 에러가 발생하기 때문에 <font color="#ffff00">pytorch cu121 dev</font> 버전으로 설치하여야합니다.

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

![[Untitled.png]]


> Google colab

```bash
import sys, os, distutils.core
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))
```

```bash
cd detectron2
```
