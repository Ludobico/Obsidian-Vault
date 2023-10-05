![[retinanet.png]]

RetinaNet은 [[Object Detection]] 을 수행하는 딥러닝 기반 알고리즘으로, 효율적이면서 강력한 성능을 보이는 모델 중 하나입니다. 이 모델은 <font color="#ffff00">Focal Loss 라고 불리는 새로운 손실 함수</font> 를 도입하여 클래스 불균형 문제를 해결하고 물체 감지 정확도를 향상시킵니다.

RetinaNet의 주요 특징 및 구성 요소는 다음과 같습니다.

1. <font color="#ffc000">Backbone Network</font>
RetinaNet은 주로 ResNet 이나 ResNeXt와 같은 pre-trained 신경망 아키텍처를 사용하여 이미지의 특징을 추출하는 데 사용합니다.

2. <font color="#ffc000">Feature Pyramid Network(FPN)</font>
FPN은 다양한 해상도의 특징 맵을 생성하여 각 개체의 크기에 따라 다양한 레벨의 특징을 활용할 수 있도록 합니다. 이것은 다양한 크기의 객체를 검출하는 데 도움이 됩니다.
![[1_aMRoAN7CtD1gdzTaZIT5gA.png]]

3. <font color="#ffc000">Anchors</font>
RetinaNet은 객체의 후보 영역을 나타내는 <font color="#ffff00">여러 크기와 종횡비를 가진 앵커(anchor) 박스</font>를 사용하여 감지를 수행합니다. 이러한 앵커는 각 위치에 대한 객체 후보를 정의하고 각 애커는 여러 크기 및 종횡비의 객체에 대응합니다.

4. <font color="#ffc000">Classification Subnet</font>
분류 서브넷은 각 앵커에 대해 <font color="#ffff00">해당 앵커가 물체 클래스에 속할 확률</font>을 예측합니다. RetinaNet은 클래스 불균형 문제를 해결하기 위해 Focal Loss를 사용하여 학습합니다.

5. <font color="#ffc000">Regression Subnet</font>
회귀 서브넷은 각 앵커에 대해 바운딩 박스를 조정하여 정확한 바운딩 박스를 예측합니다.

