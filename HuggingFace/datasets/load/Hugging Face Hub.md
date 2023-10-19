Hugging Face Hub 는 사용자가 데이터셋을 업로드하고 공유할 수 있는 플랫폼입니다.
다음은 데이터셋을 Hugging Face Hub에서 로드하는 단계에 대한 설명입니다.

1. 데이터셋 저장소 생성 및 데이터 업로드
- 먼저 Hugging Face Hub 에서 데이터셋을 공유할 데이터셋 저장소를 생성하고 원하는 데이터 파일을 해당 저장소에 업로드합니다. 이 저장소는 여러 데이터 형식 및 파일로 구성될 수 있으며, 예를 들어 csv 파일, json 파일, 이미지 파일 등을 포함합니다.

2. `load_dataset()` 함수 사용
- 데이터셋 저장소에 데이터를 업로드한 후, `load_dataset()` 함수를 사용하여 해당 데이터셋을 로드합니다. <font color="#ffff00">이 함수를 사용하려면 저장소 네임스페이스(namespace) 및 데이터셋 이름을 지정</font>합니다.

```python
from datasets import load_dataset
dataset = load_dataset("lhoestq/demo1")
```

일부 데이터셋은 Git 태그, 브랜치 또는 커밋을 기반으로 여러 버전을 가질 수 있습니다. 이러한 다양한 버전 중에서 <font color="#ffff00">특정 버전의 데이터셋을 로드</font>하려면 `revision` 파라미터를 사용하여 원하는 버전을 지정합니다.

```python
dataset = load_dataset(
  "lhoestq/custom_squad",
  revision="main"  # tag name, or branch name, or commit hash
)
```

기본적으로 데이터셋에 로딩 스크립트가 없는 경우, 데이터는 train 스플릿으로 로드됩니다. 그러나 데이터셋을 <font color="#ffff00">train, validation, test 와 같은 여러 스플릿으로 나누고 싶다면</font> `data_files` 매개변수를 사용하여 데이터 파일을 각 스플릿에 매핑할 수 있습니다.

```python
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)
```

```python
from datasets import load_dataset

# Specify the namespace and dataset name for the Hub repository
namespace = "huggingface/demo-dataset"
dataset_name = "csv_dataset"

# Specify the mapping of data files to splits using the "data_files" parameter
data_files = {
    "train": "train.csv",         # Map "train" split to "train.csv" data file
    "validation": "validation.csv", # Map "validation" split to "validation.csv" data file
    "test": "test.csv"            # Map "test" split to "test.csv" data file
}

# Load the dataset with data file mapping
dataset = load_dataset(namespace, dataset_name, data_files=data_files)

```

`load_dataset()` 함수를 사용할 때 `data_files` 또는 `data_dir` 파라미터를 사용하여 데이터셋의 특정 하위 집합을 로드할 수 있습니다. 이러한 파라미터는 데이터셋을 로드하는 기준 디렉토리에 대한 상대 경로를 받을 수 있습니다.

```python
from datasets import load_dataset

c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")

c4_subset = load_dataset("allenai/c4", data_dir="en")
```

