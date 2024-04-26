
## 행렬 곱셈의 기본 원리
---

두 행렬 $A$ 와 $B$ 가 곱해져서 행렬 $C$ 를 생성할 때, $C$ 의 각 원소는 **$A$ 의 행(row)과 $B$ 의 열(column)의 대응하는 원소들의 곱의 합** 으로 계산됩니다. 중요한 점은 행렬 $A$ 의 열의 수와 행렬 $B$ 의 행의 수가 같아야 곱셈이 가능하다는 것입니다.

예시로 행렬 $A$ 와 행렬 $B$ 를 곱하는 경우를 생각해 보겠습니다.

$$A = \begin{bmatrix}1&2&3\\4&5&6\\ \end{bmatrix}
B = \begin{bmatrix}7&8\\9&10\\11&12 \end{bmatrix}
$$

결과 행렬 $C$의 크기는 $A$ 2행 3열, $B$는 3행 2열 이므로, 결과 행렬 $C$는 2행 2열이 됩니다.

$$C = \begin{bmatrix}c_{11}&c_{12}\\c_{21}&c_{22} \end{bmatrix}$$
$$C = \begin{bmatrix}58&64\\139&154 \end{bmatrix}$$

## pytorch code
---

[[Pytorch]] 는 대규모 텐서 연산을 위한 라이브러리이며, GPU 가속을 지원합니다. 이 예제에서는 앞서 제공한 행렬 $A$와 $B$를 곱하는 과정을 구현하겠습니다.

```python
import torch

A = torch.tensor([[1,2,3], [4,5,6]])
B = torch.tensor([[7,8],[9,10],[11,12]])

C = torch.mm(A, B)

print("행렬 A:")
print(A)
print("행렬 B:")
print(B)
print("행렬 A와 B의 곱 결과 행렬 C:")
print(C)
```

```
행렬 A:
tensor([[1, 2, 3],
        [4, 5, 6]])
행렬 B:
tensor([[ 7,  8],
        [ 9, 10],
        [11, 12]])
행렬 A와 B의 곱 결과 행렬 C:
tensor([[ 58,  64],
        [139, 154]])
```

