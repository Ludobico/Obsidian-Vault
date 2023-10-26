`torch.nn.Conv3d` 는 3차원 데이터에 대한 합성곱을 수행하는 [[Pytorch]] 모듈입니다.
3차원 데이터, 특히 3D 볼륨 또는 3D 공간 데이터에서 사용되는 신경망으로 주로 다음과 같은 분야에서 적용됩니다.

1. <font color="#ffc000">의료 이미지</font>
- 의료 분야에서 CT, MRI 및 기타 의료 이미지 데이터를 다룰 때 3D [[Convolution Layers]]을 사용합니다. 3D 볼륨에서 병변을 감지하거나 의학적 이미지 분석을 수행하는 데 유용합니다.

2. <font color="#ffc000">동영상 처리</font>
- 비디오 데이터 또는 3D 동영상 시퀀스를 분석하거나 처리할 때 3D [[Convolution Layers]] 가 적합합니다. 이를 통해 동적인 시간 정보를 고려하여 <font color="#ffff00">동영상 내의 객체 검출, 추적 또는 행동 인식과 같은 작업을 수행</font>할 수 있습니다.

3. <font color="#ffc000">자연어 처리</font>
- 자연어 처리에서는 텍스트 데이터의 3D [[embedding]] 표현에 적용할 수 있습니다. 이는 3D 텐서(시간, 단어 위치, 임베딩 차원)로 텍스트를 표현하고 텍스트 간의 관계를 학습하는 데 사용될 수 있습니다.

4. <font color="#ffc000">공간 데이터</font>
- 지리 정보 시스템(GIS) 및 지리 데이터에서 공간적 정보를 분석할 때 사용합니다. 이를 통해 지형 지도, 지역 특성 및 환경 데이터를 처리하고 지리적 특성을 추출하는데 도움이 됩니다.

`torch.nn.Conv3d` 의 주요 파라미터는 다음과 같습니다.
> in_channels -> int
- 입력(3d volume)의 채널 수. 예를 들어, 흑백 이미지의 경우 채널 수는 1이며, RGB 이미지의 경우 3입니다.

> out_channels -> int
- 출력 채널 수. 이것은 컨볼루션 레이어에서 사용되는 필터(커널)의 수입니다.

> kernel_size -> int or tuple
- 커널의 크기. 이것은 커널의 윈도우 크기를 나타냅니다.

> stride -> int or tuple, (optional)
- 컨볼루션 연산의 스트라이드를 설정합니다. 스트라이드는 커널의 입력 데이터 위를 이동하는 간격을 나타냅니다. <font color="#ffc000">기본값은 1</font>이며, 다른 값으로 설정할 수 있습니다.

> padding -> int or tuple or str, (optional)
- 입력 데이터 주변에 추가되는 패딩을 설정합니다. 패딩은 컨볼루션 연산 중 입력 데이터의 크기를 유지하는 데 사용됩니다. `padding` 의 <font color="#ffc000">기본값은 0</font>이며, 필요에 따라 패딩을 추가할 수 있습니다.

> padding_mode -> str , (optional)
- 패딩 모드를 설정합니다. <font color="#ffc000">zeros, reflect, replicate, circular</font> 중 하나를 선택할 수 있습니다. 패딩 모드는 패딩 영역을 어떻게 채울지를 결정합니다. <font color="#ffc000">기본값은</font> <font color="#ffc000">zeros</font> 입니다. 

> dilation -> int or tuple, (optional)
- 커널 내 요소 사이의 간격을 설정합니다. 간격은 커널 내부의 각 원소 간의 거리를 조절하여 특정 패턴을 감지하는 데 사용됩니다. <font color="#ffc000">기본값은 1</font>이며, 다른 값을 설정할 수 있습니다.
![[Pasted image 20231026134341.png]]

> groups -> int, (optional)
- 입력 채널과 출력 채널 간의 블록 연결 수를 제어합니다. <font color="#ffff00">일반적으로 1로 설정</font>되며, 다른 값을 지정할 수 있습니다.

> bias -> bool, (optioanl)
- True로 설정하면 출려에 학습 가능한 바이어스(bias)를 추가합니다. 학습 가능한 바이어스는 컨볼루션 연산 출력에 추가적인 편향을 제공할 수 있습니다. <font color="#ffc000">기본값은 True</font>이며, False로 설정하면 바이어스를 사용하지 않도록 설정할 수 있습니다.

