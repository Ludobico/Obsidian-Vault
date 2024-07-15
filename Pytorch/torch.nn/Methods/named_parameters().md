```python
named_parameters(prefix='', recurse=True, remove_duplicate=True)
```

`named_parameters()` 메서드는 [[Pytorch]] 모델의 <font color="#ffff00">모든 파라미터를 이름과 함께 반환하는 메서드</font>입니다. 이 메서드는 모델의 각 레이어와 해당 레이어에 속한 파라미터의 이름과 값을 쌍으로 제공합니다.

일반적으로 pytorch 모델은 [[nn.Module]] 클래스를 상속하여 정의됩니다.
`named_parameters()` 메서드는 이 클래스에서 제공하는 메서드 중 하나이며, 모델 객체를 호출하여 사용할 수 있습니다.

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(10,5)
    self.fc2 = nn.Linear(5,1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
simpleModel = SimpleNN()
all_param = 0

for name, param in simpleModel.named_parameters():
  print("Parameter name : {0}".format(name))
  print("Parameter value : {0}".format(param))
  all_param += param.numel()

print("Total number of parameters : {0}".format(all_param))
```

```
Parameter name : fc1.weight
Parameter value : Parameter containing:
tensor([[-0.1093,  0.1215, -0.1604,  0.2825, -0.2866,  0.1207, -0.2746, -0.0581,
          0.0871, -0.2171],
        [ 0.0303,  0.0470,  0.0494,  0.2322,  0.1893, -0.2478, -0.2089, -0.1693,
          0.2605, -0.0393],
        [-0.0633,  0.1767,  0.2337, -0.2570,  0.0182, -0.2013,  0.2000, -0.0993,
         -0.0277,  0.3138],
        [ 0.0017, -0.0110, -0.2899, -0.3136,  0.1048, -0.0822,  0.0851, -0.1207,
          0.1678,  0.1198],
        [ 0.0609, -0.0958,  0.2589, -0.1347,  0.2834, -0.1721,  0.0231, -0.2076,
          0.0042,  0.3002]], requires_grad=True)
Parameter name : fc1.bias
Parameter value : Parameter containing:
tensor([-0.1042,  0.0043,  0.3130,  0.2036, -0.1314], requires_grad=True)
Parameter name : fc2.weight
Parameter value : Parameter containing:
tensor([[ 0.0814, -0.3271, -0.3906,  0.1661, -0.4152]], requires_grad=True)
Parameter name : fc2.bias
Parameter value : Parameter containing:
tensor([-0.3643], requires_grad=True)
Total number of parameters : 61
```

