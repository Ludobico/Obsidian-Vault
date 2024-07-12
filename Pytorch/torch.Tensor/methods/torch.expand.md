`torch.expand` 는 **주어진 텐서의 크기(size)가 1인 차원을 확장하여 더 큰 크기로 만든 새로운 뷰를 반환**합니다. 이 메서드는 새로운 메모리를 할당하지 않고, 기존 텐서의 메모리뷰를 변경합니다.

> Tensor.expand(_*sizes_) → [[Pytorch/torch.Tensor/torch.Tensor|torch.Tensor]]

>  sizes -> [[torch.size]] or int
- 확장할 크기입니다.

파이토치에서는 새로운 차원을 추가할때, 아래 코드처럼 `None` 을 사용하는 방법외에도 [[torch.unsqueeze]] 메서드를 사용하여 새로운 차원을 추가할 수 있습니다.
## example code

```python
import torch

batch = 2
num_key_value_heads = 3
slen = 4
head_dim = 5
n_rep = 2

hidden_states = torch.randn(batch, num_key_value_heads, slen, head_dim)
print(hidden_states.shape)

hidden_states = hidden_states[:,:,None, :, :]
print(hidden_states.shape)

expanded_hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
print(expanded_hidden_states.shape)
```

```
torch.Size([2, 3, 4, 5])
torch.Size([2, 3, 1, 4, 5])
torch.Size([2, 3, 2, 4, 5])
```

