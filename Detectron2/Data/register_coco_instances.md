만약 모델을 학습하는데 있어 object detection, segmentation, 또는 keypoint dataset이 [[COCO Format]] 으로 구성되어 있다면, 해당 데이터셋과 관련된 메타데이터([[MetadataCatalog]])를 쉽게 등록할 수 있습니다. 아래와 같은 코드를 사용하여 이를 수행할 수 있습니다.

```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```

만약 데이터셋이 COCO 형식으로 되어 있지만 추가적인 전처리가 필요하거나 개별 객체에 대한 사용자 지정 annotation이 있는 경우, [[load_coco_json]] 함수를 사용할 수 있습니다.

