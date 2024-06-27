`torch.Tensor.contiguous()` 는 [[Pytorch]] 에서 **텐서를 메모리 상에 연속적으로 배치된 형태로 반환**하는 메서드입니다. Pytorch에서는 텐서를 다양한 방식으로 [[torch.view]] 를 생성할 수 있으며, 이러한 뷰는 반드시 메모리에서 연속적으로 배치되지 않을 수 있습니다. 연속적이지 않은 뷰를 연속적인 형태로변환하기 위해 `contiguous()` 메서드를 사용합니다.

텐서의 연속적 배치는 **메모리 접근의 효율성을 높이고, 일부 연산은 연속적인 메모리 배치가 필요**합니다.

```python
import torch

x = torch.randn(2, 3)
print("x shape : {0}".format(x.shape))

y = x.transpose(0, 1)
print("Transposed tensor shape : {0}".format(y.shape))

print("Is transposed tensor contiguous ? {0}".format(y.is_contiguous()))

y_contiguous = y.contiguous()
print("Is contiguous tensor contiguous ? {0}".format(y_contiguous.is_contiguous()))
```

```
x shape : torch.Size([2, 3])
Transposed tensor shape : torch.Size([3, 2])
Is transposed tensor contiguous ? False
Is contiguous tensor contiguous ? True
```

