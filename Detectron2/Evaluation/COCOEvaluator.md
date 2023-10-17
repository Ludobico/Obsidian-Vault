[[Detectron2]]에서 COCO 데이터셋의 결과를 평가하고 비교하기 위해 <font color="#ffff00">COCOEvaluator</font> 를 사용할 수 있습니다. COCOEvaluator는 [[COCO Format]] 데이터셋의 검출 및 분할 결과를 기반으로 [[AP (Average Precision)]] 및 mAP 등의 평가 지표를 계산합니다.

여기서 간단한 사용 방법을 설명하겠습니다.

1. COCOEvaluator 객체 생성
COCOEvaluator 객체를 생성하고 COCO 데이터셋의 카테고리 정보를 전달합니다.
```python
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
coco_evaluator = COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_dir)
```

여기서 <font color="#ffc000">dataset_name</font> 은 coco 데이터셋의 이름, <font color="#ffc000">cfg</font>는 Detectron2의 configuration 객체이며, <font color="#ffc000">output_dir</font>은 결과를 저장할 디렉토리입니다.

전체적인 파라미터는 아래와 같습니다.

> dataset_name -> str

 - 평가할 데이터셋의 이름을 지정합니다. coco 데이터셋의 경우
- "coco_2017_val"과 같은 이름을 사용합니다.
- coco 형식의 어노테이션 파일의 경로인 "json_file" 또는 detectron2 표준 데이터셋 형식으로 데이터가 제공되어야합니다.

> tasks -> tuple[str]
- 평가할 작업을 지정합니다. <font color="#ffc000">bbox, segm, keypoints</font> 중 하나를 선택할 수 있습니다.
- default 값으로 prediction에서 이를 자동으로 예측하여 추론합니다.

> distributed -> bool
- 기본값은 <font color="#ffc000">True</font>이며, True로 설정하면 모든 랭크에서 결과를 수집하고 주 프로세스에서 평가를 수행합니다.
- <font color="#ffc000">False</font>로 설정하면 현재 프로세스에서만 결과를 평가합니다.

> output_dir -> str
- 예측 결과를 저장할 디렉토리의 경로를 지정합니다.
- "instances_predictions.pth" 및 "coco_instances_results.json" 파일이 생성됩니다.

> max_dets_per_images -> int
- 각 이미지당 최대 검출 수를 제한하는 정수값입니다. 기본값은 COCO에서는 <font color="#ffc000">100</font>입니다.

> use_fast_impl -> bool
- 빠르지만 비공식적인 방법으로 AP를 계산할지 여부를 지정합니다.
- <font color="#ffc000">True</font>로 설정하면 공식적인 COCO API와 결과가 매우 유사하게 나오지만, 여전히 논문에서 사용할 공식 API를 사용하는 것이 권장됩니다.

> kpt_oks_sigmas -> list [float]
- 키포인트 OKS(Object Keypoint Similarity)를 계산할 때 사용되는 시그마 값을 지정합니다.

> allow_cached_coco -> bool
- 이전 검증에서 캐시된 coco JSON을 사용할지 여부를 지정합니다. False로 설정하면 다른 검증 데이터를 사용할 때 유용합니다.

2. 모델 평가
모델을 사용하여 COCO 데이터셋에 대한 예측을 생성하고, 이를 COCOEvaluator에 전달하여 평가를 수행합니다.
```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Load the COCO dataset metadata
dataset_metadata = MetadataCatalog.get(dataset_name)

# Create a Predictor using the trained model
predictor = DefaultPredictor(cfg)

# Perform inference on the dataset
evaluator = COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, dataset_name)
inference_on_dataset(predictor.model, val_loader, evaluator)

```

3. 평가 결과 확인
COCOEvaluator를 통해 계산된 평가 지표를 확인할 수 있습니다.
```python
coco_evaluator.evaluate()
```

