![[Pasted image 20231011173130.png]]

Feature map은 <font color="#ffff00">이미지의 특징을 표현하는 2D 텐서</font>입니다. Feature map은 Convolutional Neural Networks(CNN)에서 사용되는 중요한 개념입니다.

Feature map은 CNN의 Convolution Layer에서 생성됩니다. Convolution Layer는 이미지를 입력으로 받아서 <font color="#ffff00">filter를 사용하여 Feature map을 생성</font>합니다.

<font color="#00b050">Feature map은 다음과 같은 특징을 가지고 있습니다.</font>

- <font color="#ffc000">크기</font>
Feature map은 이미지의 크기에 따라 크기가 결정됩니다. 일반적으로 이미지의 크기가 작으면 Feature map의 크기도 작습니다. 이미지의 크기가 크면 Feature map의 크기도 큽니다.

- <font color="#ffc000">채널</font>
Feature map은 이미지의 채널 수에 따라 채널 수가 결정됩니다. 일반적으로 이미지의 채널 수가 3이면 Feature map의 채널 수도 3입니다. 이미지의 채널 수가 1이면 Feature map의 채널 수도 1입니다.

- <font color="#ffc000">값</font>
Feature map의 값은 이미지의 특징을 나타냅니다. 예를 들어, 이미지에 있는 사람의 얼굴을 감지하는 CNN의 Feature map의 값은 얼굴의 특징을 나타냅니다.

<font color="#00b050">Feature map은 CNN에서 다음과 같은 역할을 합니다.</font>

- <font color="#ffc000">이미지의 특징 추출</font>
CNN은 이미지의 특징을 추출하여 물체를 인식합니다. Feature map은 이미지의 특징을 표현하는 2D 텐서이기 때문에, CNN은 Feature map을 사용하여 이미지의 특징을 추출할 수 있습니다.

-<font color="#ffc000"> 다양한 크기의 물체 감지</font>
CNN은 이미지를 입력으로 받아서 Feature map을 생성합니다. Feature map의 크기는 이미지의 크기에 따라 결정됩니다. 따라서, CNN은 이미지의 크기에 따라 다양한 크기의 Feature map을 생성할 수 있습니다. 이러한 Feature map을 사용하여 CNN은 다양한 크기의 물체를 감지할 수 있습니다.

- <font color="#ffc000">이미지의 세부 정보 보존</font>
CNN은 Convolution Layer에서 이미지를 입력으로 받아서 Feature map을 생성합니다. Convolution Layer는 이미지를 필터링하여 Feature map을 생성합니다. 따라서, CNN은 이미지의 세부 정보를 일부 잃을 수 있습니다. 하지만, Feature map의 크기를 조정하여 이미지의 세부 정보를 보존할 수 있습니다.