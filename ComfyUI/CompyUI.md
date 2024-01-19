
## <font color="#ffc000">Installation</font>

![[Pasted image 20240119095404.png]]

ComfyUI는 [comfyanonymous](https://github.com/comfyanonymous/ComfyUI/commits?author=comfyanonymous) 에서 만든 Image generation 라이브러리로 [Stable-Diffusion-WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 의 그래픽노드 라이브러리입니다. 

NVIDIA를 사용하는 경우 pytorch를 CUDA 버전에 맞게 설치해줍니다. 현재 CUDA 12.1 버전에 맞는 파이토치 라이브러리 설치는 아래의 명령어입니다.

<font color="#ffff00">Windows</font>
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

<font color="#ffff00">Linux</font>
```bash
pip install torch torchvision torchaudio
```
### Installation-windows

기기가 윈도우일 경우, CPU 또는 GPU로 실행시킬 수 있는 간단한 bat 파일을 제공합니다.
깃허브 레파지토리에 direct link가 있으며, 다운로드후 압축을 풀고
<font color="#ffff00">README_VERY_IMPORTANT.txt</font> 에 따라 checkpoint/models 폴더에 stable-diffution 모델을 넣어줍니다.

### Installation-Linux

깃허브 레파지 토리를 클론한뒤, SD(modles/checkpoints) 또는 VAE(models/vae) 파일을 넣습니다.
Windows 와는 다르게 리눅스에서는 bat 파일을 실행할 수 없으므로 Comfy 폴더의 <font color="#ffff00">main.py</font>를 실행합니다.

```bash
path/your/directory/ComfyUI_windows_portable/ComfyUI
```

```
main.py
```

다중 GPU를 사용하고 있고, 그 중 GPU를 선택하고 사용하려면 <font color="#ffff00">main.py</font>에 추가적인 인자를 붙여줍니다.

```
main.py --cuda-device 1
```

GPU Device 같은 경우는 [[Pytorch]] 의 `torch.cuda.current_device()` 로 확인할 수 있으며,
GPU는 0번부터시작합니다. 2대 이상 사용하는 경우 0,1,2,3 이런식으로 지정합니다.

