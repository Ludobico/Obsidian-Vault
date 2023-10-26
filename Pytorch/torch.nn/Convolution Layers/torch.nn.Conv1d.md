` torch.nn.Conv1d` 는 1차원 신호에 대해 합성곱 연산을 수행하는 [[Pytorch]]의 레이어입니다. 이 레이어는 주로 <font color="#ffff00">시계열 데이터 또는 1차원 신호(ex : 음성 데이터)와 같은 시퀀스 데이터에 적용</font>됩니다.

$$out(N_i, C_{out_j}) = bias(C_{out_j}) + \Sigma^{c_{in}-1}_{k=0} W(C_{out_j},k) \cdot input(N_i,k)$$
여기에 관한 주요 파라미터와 동작 방식에 대한 설명은 다음과 같습니다.

> in_channels -> int
- 입력 신호의 채널 수, 예를 들어 음성 데이터의 경우, 입력 특성의 수입니다.

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

```python
m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
```