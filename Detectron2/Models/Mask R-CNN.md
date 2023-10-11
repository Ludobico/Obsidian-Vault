![[1_dYb3w2iVxkN7Ifx-eA8ZRg.jpg]]

Mask R-CNN은 [[Object Detection]] 및 [[Segmentation]] 을 동시에 수행하는 딥러닝 기반의 알고리즘입니다. 이는 R-CNN 계열의 알고리즘 중 하나로, 각 물체의 바운딩 박스를 예측하는 동시에 <font color="#ffff00">픽셀 수준 </font> 에서 각 물체의 정확한 마스크를 생성합니다.

Mask R-CNN은 다음과 같은 주요 구성 요소를 포함합니다.

1. <font color="#ffc000">Backbone Network</font>
백본으로 주로 CNN 아키텍처를 사용하여 <font color="#ffff00">이미지에서 특징을 추출</font>합니다. ResNet, VGG, 또는 EfficientNet 과 같은 사전 훈련된 네트워크가 사용될 수 있습니다.

2. <font color="#ffc000">Region Proposal Network (RPN)</font>
RPN은 후보 물체의 <font color="#ffff00">바운딩 박스를 생성하여 각 후보 영역의 중요도를 점수화</font>합니다. 이러한 후보 여여들은 뒤이어 Mask R-CNN의 ROI Align 계층으로 전달합니다.

3. <font color="#ffc000">ROI Align</font>
ROI Align은 RPN에서 생성된 후보 영역을 기반으로 각 물체에 대한 정확한 [[Feature map]] 을 추출합니다. 이때, 픽셀 정화도로 마스크를 예측하기 위한 정확한 특징을 얻기 위해 ROI Pooling 대신 ROI Align을 사용합니다.

4. <font color="#ffc000">Mask Head</font>
Mask Head는 특징을 기반으로 각 <font color="#ffff00">후보 물체의 정확한 마스크를 예측</font>합니다. 주로 Fully Convolutional Network (FCN) 기반의 아키텍처가 사용되며, 각 위치의 픽셀을 물체 클래스와 관련된 마스크 값으로 예측합니다.

5. <font color="#ffc000">Classification and Bounding Box Regression Head</font>
마지막으로, 분류(classification) 및 바운딩 박스(regression)를 위한 <font color="#ffff00">두 개의 추가적인 head</font>가 있습니다. 이러한 head는 각 <font color="#ffff00">후보 물체의 클래스를 예측하고, 물체의 정확한 바운딩 박스를 조정</font>합니다.