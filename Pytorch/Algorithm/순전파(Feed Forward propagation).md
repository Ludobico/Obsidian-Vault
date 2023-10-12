![[Feed-Forward-Neural-Network.gif]]

순전파는 신경망 모델에서 입력 데이터를 <font color="#ffff00">입력층부터 출력층까지 전달하면서 예측값을 계산하는 과정</font>을 말합니다. 신경망의 순전파 단계에서는 입력 데이터가 모델을 통과하면서 각 층에서의 연산과 활성화 함수를 거쳐 출력값을 계산합니다.

순전파의 주요 단계는 다음과 같습니다.

1. <font color="#ffc000">입력 데이터</font>
모델에 입력될 데이터를 준비합니다. 이는 보통 Feature Vector로 표현되는 데이터입니다. 입력데이터는 모델의 입력층으로 들어갑니다.

2. <font color="#ffc000">가중치와 편향 적용</font>
입력 데이터가 입력층으로 들어가면, 각 연결에는 가중치(weight)가 적용됩니다. 가중치는 각 연결의 중요도를 나타내며, 학습 과정에서 업데이트됩니다. 또한, 각 층은 편향(bias)을 가질 수 있습니다. 가중치와 편향을 곱하고 더하여 각 연결에 대한 가중합을 계산합니다.

3. <font color="#ffc000">활성화 함수 적용</font>
가중합이 계산된 후, 해당 값에 [[Activation function]] 을 적용합니다. 활성화 함수는 비선형 변환을 수행하여 모델의 표현력을 증가시키고, 모델이 다양한 종류의 데이터를 모델링할 수 있도록합니다. 주로 사용되는 활성화 함수로는 sigmoid, ReLU, sofrmax 등이 있습니다.

4. <font color="#ffc000">출력 계산</font>
모든 층을 통과한 후, 마지막 출력층에서는 최종 예측값을 계산합니다. 출력층의 뉴런 수는 예측하려는 문제의 종류에 따라 달라지며, 회귀 문제인 경우 하나의 값을 출력하고, 분류 문제인 경우 클래스에 대한 확률값을 출력합니다.

5. <font color="#ffc000">손실 함수 계산</font>
예측값과 실제값을 비교하여 [[Loss function]]을 계산합니다. 손실 함수는 모델의 예측 성능을 평가하는 척도로 사용되며, 이를 최소화하는 방향으로 학습이 진행됩니다.


## <font color="#ffc000">Basic Neural Network Example</font>

```python
import torch
import torch.nn as nn
import torch.optim as optim

# FFNN 을 사용한 간단한 모델 정의
class FFNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size) -> None:
    super(FFNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out

# 임의로 생성한 훈련 데이터

x_train = torch.FloatTensor([[1,2,3,], [4,5,6,], [7,8,9]])
y_train = torch.FloatTensor([[2],[4],[6]])

# 모델, 손실함수, 최적화 함수 정의

input_size = 3
hidden_size = 100
output_size = 1

model = FFNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

  

# 모델 훈련
epochs = 1000
for epoch in range(epochs):
  model.train()
  outputs = model(x_train)
  loss = criterion(outputs, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if (epoch+1) % 100 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

  

# 모델 예측
model.eval()
with torch.no_grad():
  new_data = torch.FloatTensor([[10, 11, 12]])
  prediction = model(new_data)
  print("Prediction for [10, 11, 12]:", prediction.item())
	

```

