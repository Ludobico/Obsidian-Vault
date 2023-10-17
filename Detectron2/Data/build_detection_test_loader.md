<font color="#ffc000">build_detection_test_loader</font> 는 [[Detectron2]] 프레임워크에서 제공하는 함수로, 테스트 시 사용할 데이터로더를 구성하는 데 사용됩니다. 이 함수는 [[build_detection_train_loader]] 와 유사하지만 며 가지 차이점이 있습니다.

1. build_detection_train_loader 와 유사하지만 주로 테스트나 추론 시에 사용됩니다.

2. Default Batch Size = 1
기본 배치 사이즈는 1로 설정되어 있습니다. 이는 테스트 시에 각 배치에 하나의 이미지만이 포함된다는 것을 의미합니다.

3. <font color="#ffc000">InferenceSampler</font> 를 사용하여 샘플들의 순서를 조정합니다.

> dataset
- 테스트에 사용할 데이터셋입니다.
- 이 데이터셋은 [[DatasetCatalog]].get() 또는 <font color="#ffc000">get_detection_dataset_dicts()</font> 를 사용하여 얻을 수 있는 데이터셋 딕셔너리 리스트나 [[Pytorch]] 데이터셋입니다.

> mapper
- 데이터셋에서 샘플을 가져와 모델이 사용할 형식으로 변환하는데 사용되는 호출가능한 함수입니다.
- 기본적으로는 [[DatasetMapper]](cfg, is_train = False)를 사용합니다. cfg는 config를 나타냅니다.

> sampler
- 데이터셋에 적용할 인덱스를 생성하는 샘플러입니다.
- 기본값은 <font color="#ffc000">InferenceSampler</font>로 데이터셋을 모든 워커([[worker]])에 분할합니다.
- 만약 dataset이 iterable일 경우에는 Sampler가 <font color="#ffc000">None</font> 이어야 합니다.

> batch_size
- 생성할 데이터 로더의 배치 크기입니다.
- 기본값은 각 워커([[worker]])당 1개의 이미지로 구성된 배치입니다.

> num_workers
- 병렬 데이터 로딩 워커의 수입니다.

> collate_fn
- <font color="#ffc000">torch.utils.data.DataLoader</font>의 인수와 동일합니다.
- 기본값은 데이터를 병합하지 않고 데이터의 리스트를 반환합니다.


```python
data_loader = build_detection_test_loader(
    DatasetRegistry.get("my_test"),
    mapper=DatasetMapper(...))

# or, instantiate with a CfgNode:
data_loader = build_detection_test_loader(cfg, "my_test")
```

