
## KV Cache Explained

`KV Cache` 는 [[Transformer]] 디코더에서 key 와 value 를 **한 번 계산해 저장하고 재사용하는 구조**입니다. 이를 통해 텍스트 생성에서 **속도와 효율성을 향상** 시킵니다.

- key와 value : 이전 입력에서 계산된 후 변하지 않으며, KV Cache에 저장되어 재사용됩니다.
- Query : 현재 입력 토큰에서 매번 새로 계산됩니다.

## Example

"고양이가 박스에 앉아있다."

| 시점    | 현재 토큰 | Query     | Key/value (캐시에 저장) |
| ----- | ----- | --------- | ------------------ |
| $t=1$ | 고     | "고" 에서 생성 | "고"                |
| $t=2$ | 양     | "양" 에서 생성 | "고","양"            |
| $t=3$ | 이     | "이" 에서 생성 | "고","양","이"        |
| $t=4$ | 가     | "가" 에서 생성 | "고","양","이","가"    |
| ...   | ...   | ...       | ...(계속 누적)         |

각 시점에서

- query 는 현재 토큰에서 계속 생성됩니다.
- key와 value는 계산 후 캐시에 저장됩니다.
- 디코더는 현재 query 와 캐시된 key/value 를 사용해 다음 토큰을 예측합니다.

## example

```python
import torch
import math
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16

NUM_HEADS = 32
HEAD_DIM = 128
EMBED_DIM = NUM_HEADS * HEAD_DIM

SEQ_LEN = 512
DECODE_STEPS = 128

# projection weight 설정 
# torch.Size([4096, 4096])
W_q = torch.randn(EMBED_DIM, EMBED_DIM, device=DEVICE, dtype=DTYPE)
W_k = torch.randn(EMBED_DIM, EMBED_DIM, device=DEVICE, dtype=DTYPE)
W_v = torch.randn(EMBED_DIM, EMBED_DIM, device=DEVICE, dtype=DTYPE)

class AttentionNoKVCache:
    def __init__(self):
        self.hidden_states = []

    def step(self, x):
        self.hidden_states.append(x)
        hs = torch.stack(self.hidden_states)

        Q = hs @ W_q
        K = hs @ W_k
        V = hs @ W_v

        # 지금까지 생성된 토큰의 개수 (sequence length)
        T = hs.size(0)

        Q = Q.view(T, NUM_HEADS, HEAD_DIM)
        K = K.view(T, NUM_HEADS, HEAD_DIM)
        V = V.view(T, NUM_HEADS, HEAD_DIM)

        attn_scores = torch.einsum(
            "thd, Thd->htT", Q, K
        ) / math.sqrt(HEAD_DIM)

        atten_probs = torch.softmax(attn_scores, dim=-1)

        out = torch.einsum("htT, Thd->thd", atten_probs, V)
        return out[-1]
    
class AttentionWithKVCache:
    def __init__(self):
        self.keys = []
        self.values = []

    def step(self, x):
        q = (x @ W_q).view(NUM_HEADS, HEAD_DIM)
        k = (x @ W_k).view(NUM_HEADS, HEAD_DIM)
        v = (x @ W_v).view(NUM_HEADS, HEAD_DIM)

        self.keys.append(k)
        self.values.append(v)

        K = torch.stack(self.keys)    # (T, H, Hd)
        V = torch.stack(self.values)

        attn_scores = torch.einsum(
            "hd,Thd->hT", q, K
        ) / math.sqrt(HEAD_DIM)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        out = torch.einsum("hT,Thd->hd", attn_probs, V)
        return out
    
def benchmark(attention_cls, label):
    attn = attention_cls()
    start = time.time()

    for _ in range(DECODE_STEPS):
        x = torch.randn(EMBED_DIM, device=DEVICE, dtype=DTYPE)
        attn.step(x)

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    elapsed = time.time() - start

    print(f"{label}: {elapsed:.4f} sec")


if __name__ == "__main__":
    benchmark(AttentionNoKVCache, "NO KV CACHE")
    benchmark(AttentionWithKVCache, "WITH KV CACHE")
```

```
NO KV CACHE: 1.9059 sec
WITH KV CACHE: 0.1579 sec
```

