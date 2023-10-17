<font color="#ffc000">build_detection_train_loader</font> 함수는 [[Detectron2]] 프레임워크에서 제공하는 함수로 [[Object Detection]] 모델을 훈련하기 위한 데이터 로더를 생성하는데 사용됩니다. 이 함수는 몇 가지 기본적인 특징을 가지고 있어, 훈련에 흔히 사용되는 설정을 사용하여 데이터 로더를 생성합니다.

주요 특징 및 매개변수에 대한 설명은 다음과 같습니다.

> dataset -> list
- 훈련에 사용할 데이터셋입니다.
- 데이터셋은 [[DatasetCatalog]].get() 또는 [[get_detection_dataset_dicts]]() 를 사용하여 얻을 수 있는 데이터셋 딕셔너리의 리스트나 [[Pytorch]] 데이터셋입니다.

> mapper -> callable
- 데이터셋에서 샘플을 가져와 모델이 사용할 형식으로 변환하는데 사용되는 호출가능한 함수입니다.
- 기본적으로는 [[DatasetMapper]](cfg, is_train=True)를 사용합니다. cfg는 설정(configuration)을 나타냅니다.

> sampler -> torch.utils.data.sampler.Sampler or None
- 데이터셋에서 적용할 인덱스를 생성하는 샘플러입니다.
- 데이터셋이 map-style인 경우, 기본 샘플러는 <font color="#ffc000">TrainingSampler</font>로, 모든 워커에 걸쳐 무작위로 데이터를 셔플링합니다.
- iterable한 데이터셋의 경우에는 Sampler가 <font color="#ffc000">None</font> 이어야합니다.

> total_batch_size -> int
- 전체 배치 크기(total batch size)로, 모든 워커([[worker]])를 통틀어 적용되는 이미지의 총 수를 나타냅니다.

> aspect_ratio_grouping -> bool
- 이미지의 종횡비가 유사한 이미지들을 효율적으로 그룹화할지 여부를 나타냅니다.
- <font color="#ffc000">True</font>로 설정하면, 각 이미지에 대한 종횡비가 유사한 배치를 생성합니다. 이를 통해 학습 효율이 향상될 수 있습니다.

> num_workes -> int
- 병렬 데이터 로딩 [[worker]]의 수를 나타냅니다.

> collate_fn
- <font color="#ffc000">torch.utils.data.DataLoader</font>의 인수와 동일하며, 데이터를 배치 처리하는 방법을 결정합니다.
- 기본값은 데이터를 병합하지 않고 데이터의 리스트를 반환하며, 작은 배치 크기 및 간단한 데이터 구조에 대해 이 방법을 사용하는 것이 권장됩니다. 큰 배치 크기 및 각 샘플이 많은 작은 텐서를 포함하는 경우, 데이터 로더에서 이를 병합하는 것이 더 효율적일 수 있습니다.