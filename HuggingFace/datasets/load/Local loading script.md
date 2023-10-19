로컬 컴퓨터에 저장된 [[datasets]] 로딩 스크립트를 사용하여 데이터셋을 로드하는 경우, `load_dataset()` 함수에 다음 중 하나의 경로를 전달하여 데이터셋을 로드할 수 있습니다.

1. <font color="#ffff00">로컬 로딩 스크립트 파일의 경로</font>
- 로컬 컴퓨터에 저장된 로딩 스크립트 파일의 경로를 직접 지정할 수 있습니다. 이로써 해당 스크립트를 사용하여 데이터셋을 로드할 수 있습니다.
```python
from datasets import load_dataset

# Specify the local path to the loading script file
script_file_path = "/path/to/my_dataset_loading_script.py"

# Load the dataset using the specified loading script
dataset = load_dataset(script_file_path)

```

2. <font color="#ffff00">로딩 스크립트 파일과 동일한 이름을 가진 디렉토리의 경로</font>
- 로딩 스크립트 파일과 동일한 이름을 가진 디렉토리에 해당 스크립트 파일이 있는 경우, 해당 디렉토리의 경로를 지정하여 데이터셋을 로드할 수 있습니다.
```python
from datasets import load_dataset

# Specify the local path to the directory containing the loading script file
directory_path = "/path/to/my_dataset_script_directory"

# Load the dataset using the loading script in the specified directory
dataset = load_dataset(directory_path)

```

## <font color="#00b050"> Local and remote files</font>
---
[[datasets]] 라이브러리는 로컬 파일 및 원격 파일에서 데이터셋을 로드하는 기능을 제공합니다. 데이터셋은 주로 <font color="#ffff00">CSV,JSON,TXT</font> 또는 <font color="#ffff00">Parquet</font> 파일 형식으로 저장됩니다. `load_datasets()` 함수를 사용하여 이러한 파일 형식의 데이터셋을 로드할 수 있습니다.

CSV 파일의 경우, 다음과 같이 `load_datasets()` 함수를 사용하여 데이터셋을 로드할 수 있습니다.
```python
from datasets import load_dataset
dataset = load_dataset("csv", data_files="my_file.csv")
```

## <font color="#00b050">JSON</font>
---
JSON 파일은 다음과 같이 `load_dataset()` 함수를 사용하여 직접 로드할 수 있습니다.
```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_file.json")

```

JSON 파일은 다양한 형식을 가질 수 있지만, 가장 효율적인 형식 중 하나는 각 줄이 개별 데이터 행을 나타내는 여러 JSON 객체가 있는 형식입니다. 예를 들어
```python
{"a": 1, "b": 2.0, "c": "foo", "d": false}
{"a": 4, "b": -5.5, "c": null, "d": true}

```

JSON 파일의 다른 형식 중 하나는 중첩된 필드를 포함하는 형태입니다. 이 경우 `field` 파라미터를 사용하여 필드 이름을 지정해야 합니다. 예를 들어
```python
{"version": "0.1.0",
 "data": [{"a": 1, "b": 2.0, "c": "foo", "d": false},
          {"a": 4, "b": -5.5, "c": null, "d": true}]
}
```

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_file.json", field="data")

```
이렇게 하면 datasets 라이브러리를 사용하여 JSON 파일 형식의 데이터셋을 로드할수 있으며, JSON 파일의 형식에 따라 필요한 인수를 사용하여 로드 프로세스를 조정할 수 있습니다.

## <font color="#00b050">Pandas DataFrame</font>
---
Pandas DataFrame을 [[datasets]] 라이브러리로 로드하려면 `from_pandas()` 메서드를 사용합니다. 다음은 데이터셋을 로드하는 방법을 보여주는 예제입니다.
```python
from datasets import Dataset
import pandas as pd

# Create a Pandas DataFrame
df = pd.DataFrame({"a": [1, 2, 3]})

# Load the DataFrame into a dataset
dataset = Dataset.from_pandas(df)
```

이렇게 하면 Pandas DataFrame을 데이터셋으로 쉽게 변환할 수 있으며, 데이터셋을 다양한 형식에서 로드하고 사용하는 데 유용합니다.

다음은 `from_pandas()` 에 사용되는 파라미터에 대한 설명입니다.

> df -> pandas.DataFrame
- 로드할 데이터셋을 포함하는 판다스 데이터프레임입니다. 이 DataFrame은 데이터셋의 기반 데이터로 사용됩니다.

> features -> Features, (optional)
- 데이터셋의 특징을 나타내는 Features 객체입니다. 이 파라미터는 선택 사항이며, <font color="#ffff00">데이터셋의 특징을 정의</font>하는데 사용됩니다.

> info -> DatasetInfo, (optional)
- 데이터셋 정보를 포함하는 DatasetInfo 객체입니다. 이 정보에는 데이터셋의 설명, 인용 정보 등이 포함됩니다.

> split -> NamedSplit, (optional)
- 데이터셋의 스플릿(분할) 이름을 나타내는 NamedSplit 객체입니다. 이 파라미터를 사용하여 데이터셋을 특정 스플릿으로 로드할 수 있습니다.

> preserve_index -> bool, (optional)
- 로드된 데이터셋에서 인덱스를 보존할지 여부를 나타내는 불리언 값입니다. 이 매개변수의 기본값은 <font color="#ffc000">None</font> 이며, RangeIndex 를 제외한 인덱스는 열로 저장됩니다. <font color="#ffc000">preserve_index = True</font>로 설정하면 인덱스가 추가 열로 저장됩니다.

## <font color="#00b050">CSV</font>
---
[[datasets]] 라이브러리는 하나 또는 여러 개의 CSV 파일로 구성된 데이터셋을 읽을 수 있습니다. CSV 파일을 데이터셋으로 로드하는 방법은 다음과 같습니다.

```python
from datasets import load_dataset

# "data_files" 매개변수에 CSV 파일(들)의 경로를 리스트로 전달하여 데이터셋을 로드합니다.
dataset = load_dataset("csv", data_files=["my_file1.csv", "my_file2.csv", "my_file3.csv"])
```
