<font color="#ffff00">Detectron2</font> 는 Facebook AI Research에서 개발한 오픈 소스 프레임워크로, <font color="#ffff00">컴퓨터 비전 작업을 위한 딥러닝 프레임워크</font> 입니다. 이 프레임워크는 [[pytorch]]를 기반으로 하며, 객체 탐지, 인스턴스 분할, 
키포인트 감지 등의 컴퓨터 비전 작업을 위한 다양한 기능을 제공합니다.

<font color="#ffff00">Detectron2</font> 는 이전 버전인 Detectron 의 개선된 버전으로, 더 효율적이고 유연한 디자인과 성능향상을 제공합니다. Detectron2는 모델 개발을 위한 강력한 도구를 제공하며, 모델 아키텍처, 데이터 로딩, train, evaluation 등을 지원합니다.

Detectron2의 주요 특징은 다음과 같습니다.

### <font color="#ffc000">1. 모델 아키텍처</font>
Detectron2는 여러 가지 주요 컴퓨터 비전 작업을 위한 <font color="#ffff00">미리 학습된 모델 아키텍처</font> 를 제공합니다. 예를 들어, [[Faster R-CNN]] , [[Mask R-CNN]] , [[RetinaNet]] 등의 객체 탐지(Object detection) 모델과 [[Panoptic FPN]] , [[DeppLab]] 등의 인스턴스 분할(Instance segmentation) 모델을 사용할 수 있습니다.

### <font color="#ffc000">2. 유연성</font>
Detectron2는 유연한 구성 요소와 설정을 제공하여 사용자가 작업에 맞게 모델을 조정할 수 있습니다. 사용자는 모델의 backbone, head, loss function 등을 정의할 수 있습니다.

### <font color="#ffc000">3. 분산 훈련</font>
Detectron2는 분산 훈련을 지원하여 <font color="#ffff00">여러 GPU</font> 또는 <font color="#ffff00">여러 서버(machine)</font>를 사용하여 효율적으로 대규모 모델을 훈련할 수 있습니다.

### <font color="#ffc000">4. 간편한 데이터 로딩</font>
Detectron2는 객체 탐지와 인스턴스 분할 작업을 위한 다양한 기능을 제공합니다. 예를 들어 <font color="#ffff00">시각화, 평가 도구</font>, 모델 재사용을 위한 <font color="#ffff00">모델 체크포인트 저장 및 로드</font>등을 지원합니다.