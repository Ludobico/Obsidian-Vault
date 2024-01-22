![[Pasted image 20240122155924.png]]

[[CompyUI]] 의 Default setting입니다. stable-diffution 모델을 ckpt로 positive prompt 및 negative prompt 에 맞춰 (512,512) 사이즈의 이미지를 생성하는 json 파일입니다.

## <font color="#ffc000">Load CheckPoint</font>
---
![[Pasted image 20240122161515.png]]

CheckPoint 모델을 불러올 노드입니다. `model node`는 KSampler 노드로 연결되고 `CLIP node` 는 2개의 프롬프트 노드(positive, negative) 노드로 연결됩니다. `VAE node` 는 Ksampler를 통해 생성된 잠재벡터를 우리가 볼 수 있는 pixel image로 변환시켜주는 VAE Decode node로 연결됩니다.

