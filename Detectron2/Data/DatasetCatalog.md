<font color="#ffc000">DatasetCatalog</font> 는 [[Detectron2]] 프레임워크에서 사용되는 중요한 구성 요소로, <font color="#ffff00">데이터셋에 대한 정보를 저장하고 관리</font>하는 전역 dictionary 입니다. 이 딕셔너리는 데이터셋의 이름에 해당하는 문자열을 해당 데이터셋을 반환하는 파싱 함수에 매핑합니다. 이 파싱 함수는 데이터를 Detectron2의 데이터셋 형식은 <font color="#ffc000">list[dict]</font> 로 파싱하여 반환합니다.

여기에 <font color="#ffc000">DatasetCatalog</font> 에 대한 간단한 설명이 포함된 코드 예제가 있습니다.

```python
from detectron2.data import DatasetCatalog

# Define a parsing function for a custom dataset
def parse_custom_dataset():
    # Your logic to parse the custom dataset and return it in the desired format
    # Should return a list of dictionaries in Detectron2 Dataset format

# Register the custom dataset parsing function with a unique dataset name
DatasetCatalog.register("custom_dataset", parse_custom_dataset)

# To access the dataset later, use the registered dataset name
dataset = DatasetCatalog.get("custom_dataset")

```

### `DatasetCatalog.register`(_name_, _func_)
> name -> str
- 데이터셋을 식별하는 데 사용되는 이름입니다. 예를 들어, "coco_2014_train"과 같은 문자열로 데이터셋을 식별합니다.

> func -> callable
- 데이터셋을 반환하는 호출 가능한 함수입니다. 이 함수는 인수를 받지 않고 <font color="#ffc000">list[dict]</font> 형식의 결과를 반환해야 합니다. 이 함수를 호출하면 데이터셋을 생성하고 반환해야 합니다. 함수를 여러 번 호출하더라도 동일한 결과를 반환해야합니다.

### `DatasetCatalog.get`(_name_)
> name -> str
- 데이터셋을 식별하는 데 사용되는 이름입니다. 예를 들어, "coco_2014_train"과 같은 문자열로 데이터셋을 식별합니다.

