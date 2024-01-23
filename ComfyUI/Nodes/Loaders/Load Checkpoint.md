![[Pasted image 20240123152145.png|512]]

<font color="#ffff00">Load Checkpoint</font> 노드는 Diffusion Model을 로드하는데 사용됩니다. 이 모델은 잠재(latents) 변수를 제거하는데 사용되며, 이 노드는 적절한 VAE 및 CLIP 모델을 제공합니다.

## Inputs

- ckpt_name
	모델의 이름을 나타냅니다. 이는 로드하려는 특정 모델의 체크포인트 파일 이름 또는 식별자입니다.

## Outputs

- MODEL
	잠재 변수를 제거하는 데 사용되는 모델입니다. 이 모델은 diffusion 모델로, 이미지의 잠재 변수에서 노이즈를 제거하는 작업에 특화되어 있습니다.

- CLIP
	텍스트 [[Prompt]]를 인코딩하는 데 사용되는 CLIP 모델입니다. CLIP은 이미지 및 텍스트 간의 상호 작용을 학습한 모델로, 다양한 작업에 활용할 수 있습니다.

- VAE
	이미지를 latent space로 인코딩 및 디코딩하는 데 사용되는 VAE 모델입니다. VAE는 이미지를 latent space로 표현하고, 다시 해당 공간에서 원래 이미지를 재구성하는 데 사용됩니다.

