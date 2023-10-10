기존 `COCO2017` `ImageNet` 등 공식 coco 데이터셋이 아닌, 사용자의 커스텀 coco 데이터셋을 학습시키는 방법에 대해 서술합니다.

## <font color="#ffc000">requirements</font>
- [[Detrex]] 
설치방법은 Detrex의 [[Detrex/Installation|Installation]] 을 참조

- COCO dataset
표준 [[COCO Format]] 데이터셋이 필요합니다.


## <font color="#ffc000">step 1. check the base config file</font>
먼저 detrex 내의 학습하려는 모델이 무엇인지 확인해야 합니다.
![[Pasted image 20231010104552.png]]
2023.10 월 기준 위의 모델들이 지원됩니다.

<font color="#ffff00">Focus-DETR</font> 을 예시로 들어보면
![[Pasted image 20231010104706.png]]

위와 같은 표에<font color="#ffff00"> Name에는 config 경로</font>, <font color="#ffff00">download 에는 모델 가중치 파일</font>이 들어있습니다.

Focus-DETR-R50-4scale 의 config 파일을 예시로 들어가보면
```python

from detrex.config import get_config
from ..models.focus_detr_r50 import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
# train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "./pre-trained/resnet_torch/r50_v1.pkl"
train.output_dir = "./output/focus_detr_r50_4scale_12ep"

# max training iterations
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
```
위와 같이 되어있는데 yaml 파일이 아닌, python 파일이기때문에 모듈화 및 함수사용에 더 자유롭지만 <font color="#00b050">get_config</font> 함수를 사용함으로써 전체적인 config 구성을 확인 할 수 없다는 단점이 있습니다.

그렇기에 먼저 yaml 파일로 변환한 뒤 전체적인 구성을 파악하고 하이퍼파라미터 튜닝이 필요한 부분을 파악한뒤 다시 파이썬파일에서 그 부분을 수정해야합니다.

yaml 파일 로드 및 저장에 관해서는 [[LazyConfig]] 부분을 참고

## <font color="#ffc000">step 2. yaml load and funing</font>
```yaml
dataloader:

  evaluator: {_target_: detectron2.evaluation.COCOEvaluator, dataset_name: '${..test.dataset.names}', output_dir: ./output/dab_detr_r50_50ep_sqr}

  test:

    _target_: detectron2.data.build_detection_test_loader

    dataset: {_target_: detectron2.data.get_detection_dataset_dicts, filter_empty: false, names: coco_2017_val}

    mapper:

      _target_: detrex.data.DetrDatasetMapper

      augmentation:

      - {_target_: detectron2.data.transforms.ResizeShortestEdge, max_size: 1333, short_edge_length: 800}

      augmentation_with_crop: null

      img_format: RGB

      is_train: false

      mask_on: false

    num_workers: 4

  train:

    _target_: detectron2.data.build_detection_train_loader

    dataset: {_target_: detectron2.data.get_detection_dataset_dicts, names: coco_2017_train}

    mapper:

      _target_: detrex.data.DetrDatasetMapper

      augmentation:

      - {_target_: detectron2.data.transforms.RandomFlip}

      - _target_: detectron2.data.transforms.ResizeShortestEdge

        max_size: 1333

        sample_style: choice

        short_edge_length: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

      augmentation_with_crop:

      - {_target_: detectron2.data.transforms.RandomFlip}

      - _target_: detectron2.data.transforms.ResizeShortestEdge

        sample_style: choice

        short_edge_length: [400, 500, 600]

      - _target_: detectron2.data.transforms.RandomCrop

        crop_size: [384, 600]

        crop_type: absolute_range

      - _target_: detectron2.data.transforms.ResizeShortestEdge

        max_size: 1333

        sample_style: choice

        short_edge_length: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

      img_format: RGB

      is_train: true

      mask_on: false

    num_workers: 16

    total_batch_size: 16

lr_multiplier:

  _target_: detectron2.solver.WarmupParamScheduler

  scheduler:

    _target_: fvcore.common.param_scheduler.MultiStepParamScheduler

    milestones: [300000, 375000]

    values: [1.0, 0.1]

  warmup_factor: 0.001

  warmup_length: 0.0

  warmup_method: linear

model:

  _target_: projects.dab_detr.modeling.DABDETR

  aux_loss: true

  backbone:

    _target_: detrex.modeling.ResNet

    freeze_at: 1

    out_features: [res2, res3, res4, res5]

    stages: {_target_: detrex.modeling.ResNet.make_default_stages, depth: 50, norm: FrozenBN, stride_in_1x1: false}

    stem: {_target_: detrex.modeling.BasicStem, in_channels: 3, norm: FrozenBN, out_channels: 64}

  criterion:

    _target_: detrex.modeling.SetCriterion

    alpha: 0.25

    gamma: 2.0

    loss_class_type: focal_loss

    matcher: {_target_: detrex.modeling.HungarianMatcher, alpha: 0.25, cost_bbox: 5.0, cost_class: 2.0, cost_class_type: focal_loss_cost, cost_giou: 2.0, gamma: 2.0}

    num_classes: 80

    weight_dict: {loss_bbox: 5.0, loss_bbox_0: 5.0, loss_bbox_1: 5.0, loss_bbox_10: 5.0, loss_bbox_11: 5.0, loss_bbox_12: 5.0, loss_bbox_13: 5.0, loss_bbox_14: 5.0, loss_bbox_15: 5.0, loss_bbox_16: 5.0, loss_bbox_17: 5.0, loss_bbox_18: 5.0, loss_bbox_19: 5.0, loss_bbox_2: 5.0, loss_bbox_20: 5.0, loss_bbox_21: 5.0, loss_bbox_22: 5.0, loss_bbox_23: 5.0, loss_bbox_24: 5.0, loss_bbox_25: 5.0, loss_bbox_26: 5.0, loss_bbox_27: 5.0, loss_bbox_28: 5.0, loss_bbox_29: 5.0, loss_bbox_3: 5.0, loss_bbox_30: 5.0, loss_bbox_4: 5.0, loss_bbox_5: 5.0, loss_bbox_6: 5.0, loss_bbox_7: 5.0, loss_bbox_8: 5.0, loss_bbox_9: 5.0, loss_class: 1, loss_class_0: 1, loss_class_1: 1, loss_class_10: 1, loss_class_11: 1, loss_class_12: 1, loss_class_13: 1, loss_class_14: 1, loss_class_15: 1, loss_class_16: 1, loss_class_17: 1, loss_class_18: 1, loss_class_19: 1, loss_class_2: 1, loss_class_20: 1, loss_class_21: 1, loss_class_22: 1, loss_class_23: 1, loss_class_24: 1, loss_class_25: 1, loss_class_26: 1, loss_class_27: 1, loss_class_28: 1, loss_class_29: 1, loss_class_3: 1, loss_class_30: 1, loss_class_4: 1, loss_class_5: 1, loss_class_6: 1, loss_class_7: 1, loss_class_8: 1, loss_class_9: 1, loss_giou: 2.0, loss_giou_0: 2.0, loss_giou_1: 2.0, loss_giou_10: 2.0, loss_giou_11: 2.0, loss_giou_12: 2.0, loss_giou_13: 2.0, loss_giou_14: 2.0, loss_giou_15: 2.0, loss_giou_16: 2.0, loss_giou_17: 2.0, loss_giou_18: 2.0, loss_giou_19: 2.0, loss_giou_2: 2.0, loss_giou_20: 2.0, loss_giou_21: 2.0, loss_giou_22: 2.0, loss_giou_23: 2.0, loss_giou_24: 2.0, loss_giou_25: 2.0, loss_giou_26: 2.0, loss_giou_27: 2.0, loss_giou_28: 2.0, loss_giou_29: 2.0, loss_giou_3: 2.0, loss_giou_30: 2.0, loss_giou_4: 2.0, loss_giou_5: 2.0, loss_giou_6: 2.0, loss_giou_7: 2.0, loss_giou_8: 2.0, loss_giou_9: 2.0}

  device: cuda

  embed_dim: 256

  freeze_anchor_box_centers: true

  in_channels: 2048

  in_features: [res5]

  num_classes: 80

  num_queries: 300

  pixel_mean: [123.675, 116.28, 103.53]

  pixel_std: [58.395, 57.12, 57.375]

  position_embedding: {_target_: detrex.layers.PositionEmbeddingSine, normalize: true, num_pos_feats: 128, temperature: 20}

  select_box_nums_for_evaluation: 300

  transformer:

    _target_: projects.dab_detr.modeling.DabDetrTransformer

    decoder:

      _target_: projects.sqr_detr.modeling.DabDetrTransformerDecoder_qr

      activation: {_target_: torch.nn.PReLU}

      attn_dropout: 0.0

      embed_dim: 256

      feedforward_dim: 2048

      ffn_dropout: 0.0

      modulate_hw_attn: true

      num_heads: 8

      num_layers: 6

    encoder:

      _target_: projects.dab_detr.modeling.DabDetrTransformerEncoder

      activation: {_target_: torch.nn.PReLU}

      attn_dropout: 0.0

      embed_dim: 256

      feedforward_dim: 2048

      ffn_dropout: 0.0

      num_heads: 8

      num_layers: 6

optimizer:

  _target_: torch.optim.AdamW

  betas: [0.9, 0.999]

  lr: 0.0001

  params: {_target_: detectron2.solver.get_default_optimizer_params, base_lr: '${..lr}', lr_factor_func: !!python/name:None.%3Clambda%3E '', weight_decay_norm: 0.0}

  weight_decay: 0.0001

train:

  amp: {enabled: false}

  checkpointer: {max_to_keep: 100, period: 5000}

  clip_grad:

    enabled: true

    params: {max_norm: 0.1, norm_type: 2}

  ddp: {broadcast_buffers: false, find_unused_parameters: false, fp16_compression: false}

  device: cuda

  eval_period: 5000

  fast_dev_run: {enabled: false}

  init_checkpoint: detectron2://ImageNetPretrained/torchvision/R-50.pkl

  log_period: 20

  max_iter: 375000

  model_ema: {decay: 0.999, device: '', enabled: false, use_ema_weights_for_eval_only: false}

  output_dir: ./output/dab_detr_r50_50ep_sqr

  seed: -1

  wandb:

    enabled: false

    params: {dir: ./wandb_output, name: detrex_experiment, project: detrex}
```

detrex의 전체 config 파일입니다. 여기에서 우리가 원하는 속성은

```python
# 학습 데이터셋

print(cfg.dataloader.train.dataset.names)

# 테스트 데이터셋

print(cfg.dataloader.test.dataset.names)

# 배치 사이즈

print(cfg.dataloader.train.total_batch_size)

# 손실함수 클래스 갯수

print(cfg.model.criterion.num_classes)

# 모델 학습 클래스 갯수

print(cfg.model.num_classes)

# 학습 체크포인트

print(cfg.train.checkpointer.period)

# 학습 검증 주기

print(cfg.train.eval_period)

# iteration 갯수

print(cfg.train.max_iter)

# 베이스 모델

print(cfg.train.init_checkpoint)

# 결과저장

print(cfg.train.output_dir)
```

위와 같은데 학습 데이터셋이 이름이<font color="#ffff00"> sqr_train</font>, 검증용 데이터셋이 <font color="#ffff00">sqr_val</font>, 카테고리(클래스) 갯수가 <font color="#ffff00">98</font>, epochs를 <font color="#ffff00">100</font>으로 설정한다고 가정했을때 아래와 같이 수정됩니다.

```python
# 학습 데이터셋

cfg.dataloader.train.dataset.names="sqr_train"

# 테스트 데이터셋

cfg.dataloader.test.dataset.names="sqr_val"

# 배치 사이즈

cfg.dataloader.train.total_batch_size=1

# 손실함수 클래스 갯수

cfg.model.criterion.num_classes=98

# 모델 학습 클래스 갯수

cfg.model.num_classes=98

# 학습 체크포인트

cfg.train.checkpointer.period=18579

# 학습 검증 주기

cfg.train.eval_period=18579

# iteration 갯수

cfg.train.max_iter=185790

# 베이스 모델

cfg.train.init_checkpoint="/NAS/dlab/BGRinfo/KCJ/Detrex/model_weights/sqr_detr_0364999.pth"

# 결과저장

cfg.train.output_dir='./output/dab_detr_r50_50ep_sqr'
```

위의 수정된 정보를 토대로 다시 python 파일로 수정을 하면
```python
from detrex.config import get_config

from projects.sqr_detr.configs.models.dab_detr_r50_sqr import model

from detectron2.data.datasets import register_coco_instances

from detectron2.data import DatasetCatalog

import os

  

# GPU 전체를 사용할 예정이면 주석처리 할 것

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  

# dataset regist

dataset_names = DatasetCatalog.list()

train_dataset_name = "sqr_train"

valid_dataset_name = "sqr_val"

test_dataset_name = "sqr_test"

if train_dataset_name in dataset_names:

    pass

else:

    register_coco_instances("sqr_train",

                            {},

                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/train/segmentation.json",

                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/train/image")

  

    register_coco_instances("sqr_val",

                            {},

                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/valid/segmentation.json",

                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/valid/image")

  

    register_coco_instances("sqr_test",

                            {},

                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/test/segmentation.json",

                            "/NAS/dlab/BGRinfo/KCJ/Detectron2/dataset/08_09_merge_data_cat98_721/test/image")

  
  

dataloader = get_config("common/data/coco_detr.py").dataloader

optimizer = get_config("common/optim.py").AdamW

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

train = get_config("common/train.py").train

  

# train section

train.init_checkpoint = "/NAS/dlab/BGRinfo/KCJ/Detrex/model_weights/sqr_detr_0364999.pth"

train.output_dir = "./output/dab_detr_r50_50ep_sqr"

train.max_iter = 103100

train.eval_period = 10310

train.log_period = 20

train.checkpointer.period = 10310

train.clip_grad.enabled = True

train.clip_grad.params.max_norm = 0.1

train.clip_grad.params.norm_type = 2

train.device = "cuda"

  

# model section

model.device = train.device

model.num_classes = 98

model.criterion.num_classes = model.num_classes

  

# optimizer section

optimizer.lr = 1e-4

optimizer.betas = (0.9, 0.999)

optimizer.weight_decay = 1e-4

optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

  

# dataloader section

dataloader.train.dataset.names = train_dataset_name

dataloader.test.dataset.names = valid_dataset_name

dataloader.train.num_workers = 16

dataloader.train.total_batch_size = 6

dataloader.evaluator.output_dir = train.output_dir
```

위와 같은 설정파일이 나오게됩니다.

