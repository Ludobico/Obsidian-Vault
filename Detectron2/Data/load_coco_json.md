<font color="#ffff00">load_coco_json</font> 함수는 [[COCO Format]]의 인스턴스 형식에 따라 작성된 JSON 파일을 로드하는 역할을 합니다. 이 함수는 현재 object detection, segmentation 및 keypoint annotation 을 지원합니다.

함수의 주요 매개변수들은 다음과 같습니다.

> `detectron2.data.datasets.``load_coco_json`(_json_file_, _image_root_, _dataset_name=None_, _extra_annotation_keys=None_

- <font color="#ffc000">json_file</font>
COCO 형식의 인스턴스 정보가 담긴 JSON 파일 전체 경로

- <font color="#ffc000">image_root</font>
이미지가 존재하는 디렉토리의 경로

- <font color="#ffc000">dataset_name ( optional )</font>
데이터셋의 이름

- <font color="#ffc000">extra_annotation_keys ( optional )</font>
데이터셋에 로드해야 하는 추가적인 annotation 키의 리스트입니다. 기본적으로 <font color="#ffff00">iscrowd, bbox, keypoints, category_id, segmentation</font> 과 같은 기본 키가 로드됩니다.

> Returns

load_coco_json 함수의 반환값은 두 가지 경우에 따라 달라집니다.

1. dataset_name 이 제공된 경우
반환값은 detectron2의 표준 데이터셋 dict 형식에 따라 구성된 사전들의 리스트입니다. 각 dict는 하나의 데이터 샘플을 나타내며, Detectron2 표준 데이터셋 형식에 따라 필드가 구성되어 있습니다. 이 경우, category_ids 는 연속적인 범위로 매핑되어 있으며 이 또한 Detectron2 표준 형식입니다.

2. dataset_name이 None일 경우
반환값은 COCO 데이터셋의 표준 형식이 아닐 수도 있습니다. 이 경우, 데이터셋 이름이 제공되지 않았으므로 각 데이터 샘플에 대한 표준 포맷이 적용되지 않았을 가능성이 있습니다.

