
## Linux

리눅스 기준으로 프레임워크와 의존성을 설치한뒤 [[Kohya_ss]] 폴더로 이동한 뒤, 아래의 명령어로 GUI 환경을 실행시킵니다.

```
./gui.sh
```

<font color="#ffff00">dataset</font> 폴더에 아래와 같이 폴더가 있는지 확인하고, 없으면 폴더를 생성합니다.

![[Pasted image 20240402095034.png]]

<font color="#ffff00">images</font> 폴더에 학습을 진행할 이미지를 넣어주는데 **폴더 이름은 (repeat) name**  이런형식으로 얼마나 반복할지를 명시하여 넣어야 합니다.

![[Pasted image 20240402095155.png]]

![[Pasted image 20240402095303.png]]

폴더에는 **하나의 이미지 파일에 대응하는 텍스트파일**이 존재합니다. 이미지 파일에는 LORA를 진행할 이미지 파일을 넣어주고, 텍스트 파일에는 그 이미지를 CLIP으로 설명할 텍스트파일과 LORA keyword를 표시합니다. 예를 들어

![[Pasted image 20240402095456.png]]

위와 같은 이미지와 대응하는 텍스트로는

```
dying light, a picture of a cross with two crossed pens
```

위와 같이 lora keyword에 해당하는 것으로는 dying light가 입력되었고, 이미지를 설명하는 텍스트가 입력되어있습니다.

만일 이미지가 존재하고 텍스트가 존재하지않는다면, <font color="#ffff00">Utillities</font> 의 <font color="#ffff00">BLIP Captioning</font> 기능을 사용하여 **각 이미지의 설명을 텍스트로 변환하는 기능**을 사용합니다.

## Train

![[Pasted image 20240402102750.png]]

LORA 모델을 train 하려면 <font color="#ffff00">LORA</font> 의 <font color="#ffff00">Training </font> 기능을 사용하여 학습을 진행합니다. 여기서 각 모델의 output 폴더, 학습할 이미지 폴더, log 폴더등을 지정한뒤,<font color="#ffff00"> Parameters</font> 섹션에서 파인튜닝을 진행한뒤 

<font color="#ffff00">Start training</font> 버튼을 눌러 학습을 진행합니다.