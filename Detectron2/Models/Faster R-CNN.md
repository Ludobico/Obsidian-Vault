![[faster r-cnn.png]]

Faster R-CNN은 컴퓨터 비전 분야에서 객체 탐지([[Object Detection]])와 관련된 최신 딥러닝 기반의 알고리즘 중 하나입니다. R-CNN(Regions with Convolutional Neural Network) 계열의 알고리즘 중 하나로, 물체 감지를 위한 효과적인 방법을 제공합니다.

Faster R-CNN은 물체 감지를 두 단계로 나누어 수행합니다. 먼저, <font color="#ffff00">물체가 있을 것으로 예상되는 영역을 식별</font>하고, <font color="#ffff00">각 영역에 대해 물체를 식별</font>합니다.

## <font color="#ffc000">Region Proposal Network(RPN)</font>
Faster R-CNN의 첫 번째 단계는 RPN을 사용하여 <font color="#ffff00">이미지 내에 물체가 존재할 것으로 예상되는 후보 영역(Region of Interest, ROI)</font> 을 제안하는 것입니다. 이 단계에서는 각 위치에 대해 여러 크기와 종횡비를 가진 후보 영역을 제안하는데, 이러한 제안된 영역들은 물체가 있을 가능성이 높은 위치를 나타냅니다.

![[1_FifNx4NCyynAZqLjVtB5Ow.png]]


## <font color="#ffc000">ROI Pooling 및 Classifier/Regressor</font>
두 번째 단계에서는 RPN을 통해 제안된 각 ROI에 대해 특징을 추출하고, 이를 이용하여<font color="#ffff00"> 물체를 식별</font>하는 분류(Classifier) 및 <font color="#ffff00">바운딩 박스를 조정</font>하는 회귀(Regressor)를 적용합니다. 이 단계에서는 ROI Pooling을 사용하여 각 ROI에 대한 고정된 크기를 추출하고, 이 특징을 입력으로 사용하여 물체를 식별하고 바운딩 박스를 조정합니다.

Faster R-CNN은 전체 네트워크를 통합하여 [[end-to-end]] 방식으로 학습되며, 학습 데이터에 대한 물체 감지의 정확도를 향상시킵니다. 이러한 두 단계 접근 방식을 통해 정확하고 효율적인 물체 감지를 달성할 수 있으며, 빠른 속도로 처리할 수 있어 Faster R-CNN이라고 불립니다.