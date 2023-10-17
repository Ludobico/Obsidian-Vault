<font color="#ffc000">get_detection_dataset_dicts</font> 는 [[Detectron2]] 프레임워크에서 제공하는 함수로 [[Object Detection]] , [[Segmentation]] 및 [[Semactic segmentation]] 작업에 사용할 수 있는 <font color="#ffff00">데이터셋을 로드하고 준비</font>하는 역할을 합니다.

> naems -> str or list[str]
- 데이터셋의 이름이나 데이터셋 이름의 리스트입니다. 이는 [[DatasetCatalog]]에 등록된 데이터셋의 이름입니다.

> filter_empty -> bool
- <font color="#ffc000">True</font>로 설정하면, <font color="#ffff00">객체 어노테이션이 없는 이미지를 필터링하여 반환</font>합니다.

> min_keypoints -> int
- 키포인트 수가 이 값보다 작은 데이터를 필터링하여 반환합니다. 0으로 설정하면 키포인트 수로 필터링을 수행하지 않습니다.

> proposal_files -> list[str]
- proposal 파일의 경로 리스트를 제공하여 특정 데이터셋에 대한 proposal을 추가할 수 있습니다.

> check_consistency -> bool
- <font color="#ffc000">True</font>로 설정하면, 데이터셋의 메타데이터가 일관성 있는지 확인합니다.

