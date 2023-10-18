<font color="#ffc000">DatasetMapper</font> 는 [[Detectron2]] 프레임워크에서 사용되는 클래스로, <font color="#ffff00">데이터셋 딕셔너리를 Detectron2 모델에서 사용하는 형식으로 변환</font>하는 역할을 수행하는 클래스입니다.

기본적으로 DatasetMapper 는 사용자가 제공한 데이터셋 딕셔너리를 train 데이터로 매핑하기 위해 사용되는 객체입니다. 그러나 사용자가 필요에 따라 데이터셋을 사용자 지정 방식으로 변환하기 위해 자체 DatasetMapper 를 구현할 수도있습니다.

주요 기능으로는 다음과 같습니다.

1. <font color="#ffff00">파일에서 이미지 읽기</font>
- 데이터셋 딕셔너리에 있는 file_name 필드에서 이미지를 읽습니다.

2. <font color="#ffff00">이미지 및 어노테이션에 대한 크롭/가하학적 변환 적용</font>
- 이미지와 어노테이션에 대해 크롭 및 기하학적 변환을 적용합니다. 이는 <font color="#ffff00">데이터 증강(data augmentation) 단계로서, 모델 학습을 향상시키고 일반화를 높이기 위해 수행</font>됩니다.

3. <font color="#ffff00">데이터 및 어노테이션을 Tensor 및 Instances 형식으로 준비</font>
- 데이터 및 어노테이션을 텐서 및 인스턴스 형식으로 변환하여 모델이 사용할 수 있도록 준비합니다.


>is_train -> bool
- 훈련(training) 또는 추론(inference)에 사용될지 여부를 나타내는 불리언 값입니다.

> augmentations -> list
- 적용할 증강 또는 변환 리스트입니다.

> image_format -> str
- [[detection_utils]].read_images() 에서 지원하는 이미지 형식입니다.

> use_instance_mask -> bool
- 인스턴스 세그멘테이션 어노테이션을 처리할지 여부를 나타내는 불리언 값입니다.

> use_keypoint -> bool
- 키포인트 어노테이션을 처리할지 여부를 나타내는 불리언 값입니다.

> instance_mask_format -> str
- 키포인트 어노테이션을 처리할지 여부를 나타내는 불리언 값입니다.

> precomputed_proposal_topk -> int
- 지정된 경우, 데이터셋 딕셔너리에서 사전 계산된 제안(proposal)을 로드하고 각 이미지에 대해 상위 k개의 제안을 유지합니다.

> recompute_boxes -> bool
- 바운딩 박스 어노테이션을 계산된 인스턴스 마스크 어노테이션에서 정확한 바운딩 박스로 덮어쓸지 여부를 나타내는 불리언 값입니다.

