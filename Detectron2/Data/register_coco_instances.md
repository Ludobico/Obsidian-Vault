만약 모델을 학습하는데 있어 object detection, segmentation, 또는 keypoint dataset이 [[COCO Format]] 으로 구성되어 있다면, 해당 데이터셋과 관련된 메타데이터([[MetadataCatalog]])를 쉽게 등록할 수 있습니다. 아래와 같은 코드를 사용하여 이를 수행할 수 있습니다.

```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```

만약 데이터셋이 COCO 형식으로 되어 있지만 추가적인 전처리가 필요하거나 개별 객체에 대한 사용자 지정 annotation이 있는 경우, [[load_coco_json]] 함수를 사용할 수 있습니다.

## 예시 등록
---
```python
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

dataset_names = DatasetCatalog.list()
train_dataset_name = "sqr_train"
valid_dataset_name = "sqr_val"
test_dataset_name = "sqr_test"
if train_dataset_name in dataset_names:
    pass
else:
    register_coco_instances(train_dataset_name,
                            {},
                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/train/segmentation.json",
                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/train/image")

    register_coco_instances(valid_dataset_name,
                            {},
                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/valid/segmentation.json",
                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/valid/image")

    register_coco_instances(test_dataset_name,
                            {},
                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/test/segmentation.json",
                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/test/image")
```