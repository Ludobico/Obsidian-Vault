- [[#build_rotary_frequencies|build_rotary_frequencies]]
- [[#apply_rope|apply_rope]]


![[Pasted image 20250901112407.png]]

<font color="#ffff00">Rotary Positional Embedding (RoPe)</font> 는 [[Transformer]] 모델에서 **토큰의 위치 정보를 인코딩하는 방법 중 하나**입니다. 기존의 절대 위치 임베딩 이나 상대 위치 임베딩 대신, RoPE는 **회전행렬(Rotation Matrix)을 사용해 위치 정보를 표현**합니다. 이는 다음과 같은 특징을 가집니다.

- 회전 기반 위치 인코딩 : 각 토큰의 임베딩 벡터를 2D 평면에서 회전시키는 방식으로 위치 정보를 부여합니다. 이를 통해 위치 간 상대적 거리가 자연스럽게 모델에 반영됩니다.

- 차원 쌍(pariwise) 처리 : 임베딩 차원을 쌍으로 묶어 2D 회전을 적용합니다. 즉 차원 $d$ 가 짝수여야 하며, 각 쌍 ($x_{2i}$, $x_{2i+1}$ )에 대해 회전 변환을 수행합니다.

- RoPE는 주기적인 사인/코사인 함수를 사용해 위치를 인코딩하므로, 각 시퀀스에서도 일반화가 잘 됩니다.

- RoPE는 절대 위치보다 상대적 위치 정보를 더 잘 반영하며, 특히 어텐션 메커니즘에서 쿼리와 키의 [[dot product]] 에 위치 정보를 통합합니다.

RoPE의 핵심 아이디어는 위치 $m$ 에 있는 토큰의 임베딩 $x_m$ 에 대해, 위치 정보를 회전 행렬로 변환하여 새로운 임베딩 $x'_m$ 를 생성하는 것입니다. 이는 다음과 같이 표현됩니다.

$$
x'_m = R(m, \theta) \cdot x_m
$$

여기서 $R(m, \theta)$ 는 위치 $m$ 과 주파수 $\theta$ 에 기반한 회전 행렬입니다.

코드는 다음과 같이 나타냅니다.

```python
import torch

# ROPE
def build_rotary_frequencies(dim_head : int, max_seq_len : int, base : float = 10000.0, device = None):
    assert dim_head % 2 == 0 # head_dim must be even for RoPE
    
    device = device or "cpu"
    half_dim = dim_head // 2

    # theta : [half_dim]
    theta = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    # positions : [seq_len, 1]
    seq = torch.arange(max_seq_len, device=device).float()
    freqs = torch.einsum("s,d -> sd", seq, theta)

    # interleave to shape [seq, dim_head] as [cos(x0), cos(x1), ..., sin(x0), sin(x1), ...]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # expand to [1, 1, seq, half_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    return cos, sin

def apply_rope(x, cos, sin):
    """
    x: [B, H, S, D], D even.
    cos/sin: [1,1,S, D/2] (we will split x to half halves)
    Returns rotated x.
    """

    B, H, S, D = x.shape
    x_ = x.view(B, H, S, D // 2, 2) # pair last dim: (even, odd)
    x_even = x_[..., 0]
    x_odd = x_[..., 1]

    # rotation: (x_even, x_odd) -> (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(B, H, S, D)
    return x_rot

```

## build_rotary_frequencies

이 함수는 RoPE의 **회전 주파수를 생성**합니다. 입력으로 헤드차원(`dim_head`), 최대 시퀀스 길이(`max_seq_len`), 기본 주파수(`base`), 디바이스(`device`) 를 받습니다.

주파수 $\theta_i$ 는 다음과 같이 정의됩니다.

$$
\theta_i = \text{base}^{-\frac{2i}{d}}, i = 0,1,...,\frac{d}{2}-1
$$

- $d$ : 헤드 차원
- base : 기본 주파수 (보통 10000)
- $i$ : 차원 인덱스

위치 $m$ 과 주파수 $\theta_i$ 를 곱해 회전 각도를 계산합니다.

$$
\phi_{m,i} = m \cdot \theta_i
$$

이를 바탕으로 코사인과 사인 값을 계산합니다.

$$
\cos(\phi_{m,i}), \sin(\phi_{m,i})
$$

## apply_rope

이 함수는 입력 임베딩에 RoPE를 적용하여 위치 정보를 반영한 새로운 임베딩을 생성합니다.

입력:
- x: \[B, H, S, D\] 형태의 입력 텐서(B: 배치 크기, H: 헤드 수, S: 시퀀스 길이, D: 헤드 차원).
- cos, sin: \[1, 1, S, D/2\] 형태의 회전 주파수 텐서.

차원 $d$ 에 대해, 회전 행렬 $R(m, \theta)$ 는 블록 대각 행렬 형태로 구성됩니다.

$$
R(m, \theta) = \begin{bmatrix} R_0 & 0 & \cdots & 0 \\ 0 & R_1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & R_{\frac{d}{2}-1} \end{bmatrix}​​​
$$

여기서 각 블록 $R_i$ 는 2x2 회전 행렬입니다.

$$
R_i(m, \theta_i) \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
$$


입력 벡터 $x_m = [x_0, x_1, ..., x_{d-1}]$ 은 쌍 ($x_{2i}, x_{2i+1}$) 으로 나뉘어 각 블록에 대해 회전이 적용됩니다.

$$
\begin{bmatrix} x'_{2i} \\ x'_{xi+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{xi+1} \end{bmatrix}
$$

이를 풀면

$$
\begin{aligned}
x'_{2i} = x_{2i}\cos(m\theta_i) - x_{2i+1}\sin(m\theta_i) \\
x'_{2i+1} = x_{2i}\sin(m\theta_i) + x_{xi+1}\cos(m\theta_i)
\end{aligned}
$$

이렇게 됩니다.

