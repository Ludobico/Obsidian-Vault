`torch.nn.ConvTranspose2d` 는 2차원 이미지 또는 데이터에 대한 전치 합성곱 연산을 수행하는 [[Pytorch]] 의 클래스입니다. 이 클래스는 주로 이미지 처리 및 컨볼루션 신경망(CNN)의 [[upsampling]] 레이어를 구성하는 데 사용됩니다. 전치 합성곱 연산은 입력 데이터의 공간 해상도를 높이는데 유용하며, 예를 들어 이미지 [[Segmentation]] , 이미지 생성(GANs), 슈퍼 해상도 및 이미지 복원 작업에서 널리 사용됩니다.

`torch.nn.ConvTranspose2d` 클래스는 다음과 같은 주요 매개변수와 기능을 가집니다.

> in_channels -> int
- 입력의 채널 수 입니다. 이것은 입력 데이터의 차원을 나타냅니다.

> out_channels -> int
- 합성곱 연산을 통해 생성된 출력 채널의 수입니다. 이것은 필터 또는 커널의 개수를 나타냅니다.

> kernel_size -> int ot tuple, (optional)
- 커널의 크기 또는 높이를 정의합니다. 이것은 합성곱 커널의 윈도우 크기를 나타냅니다.

> stride -> int or tuple, (optional), default = 1
- 합성곱 연산에서 필터 이동 간격을 나타냅니다. 더 큰 값은 출력 크기를 줄입니다.

> padding -> int or tuple, (optional) default = 0
- 입력에 대한 제로 패딩(zero-padding)을 설정합니다. 패딩은 커널의 크기 및 dilation에 따라 조절됩니다.

> output_padding -> int or tuple, (optional) default = 0
- 츌력에 대한 추가적인 크기 조정을 제어합니다. 이것은 디<font color="#ffff00">컨볼루션 연산 이후에 출력 크기를 더 확장하는 데 사용</font>됩니다.

> groups -> int, (optional) default = 1
- 입력 채널과 출력 채널 간의 블록된 연결 수를 설정합니다. 이것은 그룹 합성곱 연산을 나타냅니다.

> bias -> bool, (optional) default = True
- True로 설정하면, 모델은 출력에 bias를 추가합니다. 편향은 모델이 데이터에 더 적응하도록 도와줍니다.

> dilation -> int or tuple, (optional) default = 1
- 커널 내 요소 사이의 간격을 설정합니다. 이것은 증식된 합성곱(dilated convolution)을 수행하며, 주어진 간격으로 원소를 건너뛰어 데이터 처리를 제어합니다.
![[Pasted image 20231026134341.png]]

```python
# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)
output = m(input)
# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
h.size()
output = upsample(h, output_size=input.size())
output.size()
```

