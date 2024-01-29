```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
```

[[Pytorch]] 에서는 임베딩 벡터를 사용하는 방법이 크게 두 가지가 있습니다.<font color="#ffff00"> embedding layer</font> 를 만들어 훈련 데이터로부터 처음부터 임베딩 벡터를 학습하는 방법과 미리 사전에 훈련된 <font color="#ffff00">pre-trained word embedding</font> 들을 가져와 사용하는 방법입니다.

> num_embeddings -> int
- 임베딩 딕셔너리의 크기입니다. 즉, 사용할 고유한 토큰의 총 개수를 나타냅니다.

> embedding_dim -> int
- 각 임베딩 벡터의 차원 크기입니다. 이 값의 임베딩 공간의 차원을 결정합니다. 예를 들어, 100차원의 임베딩 벡터를 사용하려면 `embedding_dim=100` 으로 설정합니다.

> padding_idx -> int, optional
- [[typing]] 의 optional 로 지정할 경우, padding_idx 위치의 토큰에 대한 임베딩 벡터는 학습 중에 업데이트되지 않습니다. 일반적으로 패딩 토큰을 표시하는 데 사용되며, 기본적으로 0으로 설정됩니다.

> max_norm -> float, optional
- 주어진 경우, 임베딩 벡터의 L2 Norm이 max_norm보다 큰 경우 임베딩 벡터를 재조정합니다. 이를 통해 임베딩 값의 크기를 제한할 수 있습니다.

> norm_type -> float, optional
- max_norm 옵션에서 사용되는 p-norm의 p 값을 지정합니다. 기본값은 2로, 이는 L2 Norm을 사용함을 의미합니다.

> scale_grad_by_freq -> bool, optional
- 주어진 경우, 미니 배치 내 단어의 빈도에 따라 gradient를 스케일합니다. 빈도가 낮은 단어의 gradient는 더 크게 스케일됩니다. 기본값은 False 입니다.

> sparse -> bool, optional
- True로 설정할 경우, 가중치 행렬에 대한 gradient가 희소 텐서로 반환됩니다. 이는 메모리 효율성을 높일 수 있습니다.

## <font color="#ffc000">임베딩 층은 룩업 테이블이다.</font>
---
임베딩 층의 입력으로 사용하기 위해서<font color="#ffff00"> 입력 시퀀스의 각 단어들은 모두 정수 인코딩</font>이 되어있어야 합니다.

> 어떤단어 -> 단어에 부여된 고유한 정수 값 -> 임베딩 층 통과 -> 밀집 벡터

임베딩 층은 입력 정수에 대해 <font color="#00b050">밀집 벡터(dense vector)</font> 로 맵핑하고 이 밀집 벡터는 인공 신경망의 학습 과정에서 가중치가 학습 되는 것과 같은 방식으로 훈련됩니다. 훈련 과정에서 단어는 모델이 풀고자하는 작업에 맞는 값으로 업데이트됩니다. 그리고 <font color="#ffff00">이 밀집 벡터를 임베딩 벡터라고 부릅니다</font>.

정수를 임베딩 벡터로 맵핑한다는 것은 어떤 의미일까요? 특정 단어와 맵핑되는 정수를 인덱스로 가지는 테이블로부터 임베딩 벡터 값을 가져오는 룩업 테이블이라고 볼 수 있습니다. 그리고 이 테이블은 단어 집합의 크기만큼의 행을 가지므로 모든 단어는 고유한 임베딩 벡터를 가집니다.

![[Pasted image 20240129105525.png]]

위의 그림은 단어 great이 정수 인코딩 된 후 테이블로부터 해당 인덱스에 위치한 임베딩 벡터를 꺼내오는 모습을 보여줍니다. 위의 그림에서 임베딩 벡터의 차원이 4로 설정되어져 있습니다. 그리고 단어 great은 정수 인코딩 과정에서 1,918의 정수로 인코딩이 되었고 그에 따라 단어 집합의 크기마큼의 행을 가지는 테이블에서 인덱스 1,918번에 위치한 행을 단어 great의 임베딩 벡터로 사용합니다. 이 임베딩 벡터는 모델의 입력이 되고, [[Backward propagation]] 과정에서 단어 greate의 임베딩 벡터값이 학습됩니다.

룩업 테이블의 개념을 이론적으로 우선 접하고, 처음 [[Pytorch]] 를 배울 때 어떤 분들은 임베딩 층의 입력이 원-핫 벡터가 아니어도 동작한다는 점에 헷갈려 합니다. 파이토치는 단어를 정수 인덱스로 바꾸고 원-핫 벡터로 한번 더 바꾸고나서 임베딩 층의 입력으로 사용하는 것이 아니라, <font color="#ffff00">단어를 정수 인덱스로만 바꾼채로 임베딩 층의 입력으로 사용해도 룩업 테이블 된 결과인 임베딩 벡터를 리턴</font>합니다.

룩업 테이블 과정을 nn.Embedding()을 사용하지 않고 구현해보면서 이해해보겠습니다.
우선 임의의 문장으로붜 단어 집합을 만들고 각 단어에 정수를 부여합니다.

```python
train_data = 'Blending is all you need'

# 중복을 제거한 단어들의 집합인 단어 집합 생성
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

print(vocab)
```

```
{'need': 2, 'is': 3, 'you': 4, 'Blending': 5, 'all': 6, '<unk>': 0, '<pad>': 1}
```

이제 단어 집합의 크기를 행으로 가지는 임베딩 테이블을 구현합니다. 단, 여기서 임베딩 벡터의 차원은 3으로 정했습니다.

```python
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9]])
```

이제 임의의 문장 `you need to run` 에 대해서 룩업 테이블을 통해 임베딩 벡터들을 가져와보겠습니다.

```python
import torch

train_data = 'Blending is all you need'

# 중복을 제거한 단어들의 집합인 단어 집합 생성
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9]])

sample = 'you need to run'.split()
idxes = []

# 각 단어를 정수로 변환
for word in sample:
  try:
    idxes.append(vocab[word])
  # 단어 집합에 없는 단어일 경우 <unk>로 대체
  except KeyError:
    idxes.append(vocab['<unk>'])

idxes = torch.LongTensor(idxes)

# 각 정수를 인덱스로 임베딩 테이블에서 값을 가져옴
# idxes에 [2,5,7] 이 있다면 embedding_table에서 인덱스로 2,5,7에 해당하는 행을 선택하여
# 해당 행에 대한 임베딩 벡터를 반환
lookup_result = embedding_table[idxes, :]
print(lookup_result)
print("--------------------------------------")
print(idxes)
print("--------------------------------------")
print(embedding_table)
```

```
tensor([[0.2000, 0.9000, 0.3000],
        [0.1000, 0.8000, 0.9000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
--------------------------------------
tensor([2, 6, 0, 0])
--------------------------------------
tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.2000, 0.9000, 0.3000],
        [0.1000, 0.5000, 0.7000],
        [0.2000, 0.1000, 0.8000],
        [0.4000, 0.1000, 0.1000],
        [0.1000, 0.8000, 0.9000]])
```

## <font color="#ffc000">use Embedding Layer</font>
---
이제 `nn.Embedding()`으로 사용할 경우를 봅시다. 우선 전처리는 동일한 과정을 거칩니다.

```python
import torch

train_data = 'Blending is all you need'

# 중복을 제거한 단어들의 집합인 단어 집합 생성
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
```

이제 `nn.Embedding()` 을 사용하여 학습가능한 임베딩 테이블을 만듭니다.

```python
import torch
import torch.nn as nn

train_data = 'Blending is all you need'

# 중복을 제거한 단어들의 집합인 단어 집합 생성
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3, padding_idx=1)
print(embedding_layer.weight)
```

```
Parameter containing:
tensor([[ 0.4891,  0.2660, -0.7561],
        [ 0.0000,  0.0000,  0.0000],
        [-1.1794, -1.0988, -1.0689],
        [-0.4006,  1.3217,  0.7954],
        [-0.4366,  1.1717,  1.3234],
        [-2.1029, -0.8586,  1.6083],
        [ 1.2279,  1.8150,  2.6248]], requires_grad=True)
```

`nn.Embedding`은 크게 두 가지 인자를 받는데 각각 num_embeddings와 embedding_dim 입니다.

- num_embeddings : 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기입니다.
- embedding_dim : 임베딩 할 벡터의 차원입니다. 사용자가 정해주는 하이퍼파라미터입니다.
- padding_idx : 패딩을 위한 토큰의 인덱스를 알려줍니다.

앞선 예제와 마찬가지로 단어 집합의 크기의 행을 가지는 임베딩 테이블이 생성되었습니다.

