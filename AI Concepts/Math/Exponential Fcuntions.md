- [[#지수함수 그래프|지수함수 그래프]]
- [[#지수함수의 평행 이동과 대칭 이동|지수함수의 평행 이동과 대칭 이동]]
	- [[#지수함수의 평행 이동과 대칭 이동#지수함수의 평행 이동|지수함수의 평행 이동]]
	- [[#지수함수의 평행 이동과 대칭 이동#지수함수의 대칭 이동|지수함수의 대칭 이동]]


지수함수는 지수에 미지수 $x$ 가 있는 함수, 즉 $f(x) = a^x (a > 0, a \ne 1)$ 형태로 나타낼 수 있는 함수입니다. 다음과 같이 표현할 수 있다면 지수함수입니다.

$$f(x) = a^x ( a>0, a\ne 0)$$
$a$ 는 <font color="#ffff00">밑</font>이라 하고, $x$ 는 <font color="#ffff00">지수(변수)</font>라고 합니다.

> 밑은 양의 정수여야합니다. 즉 0보다 큰 양수고, 1이면 안됩니다. a = 1 이 되면 어떤 값이 들어가도 $f(x)$ 는 항상 1이 되기 때문입니다. 따라서 다음과 같이 두 종류로 나눌 수 있습니다.
> 
> 0 < a < 1
> a > 1

## 지수함수 그래프
---
$y = a^x$ 가 지수함수라고 했습니다. 앞서 $a$ 가 0보다 크다고 했으므로 임의의 숫자 2를 대입한 후 $x$ 의 변화에 따른 그래프를 그려보겠습니다. 아래는 $y = a^x$ 에서 $a$ 가 2일 때 $x$ 값에 따른 $y$ 값의 변화를 나타낸 것입니다.

```python
x = [-2, -1, 0, 1, 2]
a = 2
for i in x:
  y = a**i
  print(y)
```

```
0.25
0.5
1
2
4
```

지수가 커지면 $y$ 값도 커지고, 지수가 작아지면 $y$ 값도 작아지지만 0보다는 큽니다. 즉, 지수 $x$ 가 커지면 $y$ 도 커지기 때문에 다음과 같이 <font color="#ffff00">오른쪽 위로 향하는 그래프</font>가 됩니다. 또 지수 $x$ 가 작아지면 $y$ 도 작아지기 때문에<font color="#ffff00"> 0에 한없이 가까워지는 그래프</font>가 됩니다. 

![[Pasted image 20240228135631.png]]
> a > 1 일 때 지수함수


이번에는 $0 < a < 1$ 인 경우를 살펴볼게요. 임의의 수 $\frac{1}{2}$ 를 a에 대입한 후 $x$ 를 변화시켜 보겠습니다.

```python
from fractions import Fraction

x = [-2, -1, 0, 1, 2]
a = Fraction(1, 2)
for i in x:
  y = a**i
  print(y)
```

```
4
2
1
1/2
1/4
```

즉, 지수가 작아질수록 $y$ 값은 커지고, 지수가 커질수록 $y$ 값은 작아집니다. 지수 $x$ 가 커지면 $y$ 는 작아지므로 다음과 같이 <font color="#ffff00">오른쪽 아래</font>로 향합니다.

![[Pasted image 20240228140309.png]]
> 0 < a < 1 일 때 지수함수

이때 밑이 역수인 두 지수함수는 $y$ 축에 대해 대칭입니다. 예를 들어

$$2^{-2} = \frac{1}{4}$$
과
$$(\frac{1}{2})^2 = \frac{1}{4}$$
의 값이 같습니다. 다시 말해 밑이 역수일 때 지수인 $x$ 의 부호가 반대이면 $y$ 값이 같습니다. 그리고 이때 두 지수함수는 $y$ 축에 대칭입니다. 

## 지수함수의 평행 이동과 대칭 이동
---

### 지수함수의 평행 이동

지수함수 $y = a^x$ 를 평행 이동하면 어떻게 표현할 수 있을까요? 평행 이동이기 때문에 <font color="#ffff00">그래프 모양은 바뀌지 않으면서 위치만 바뀔 것</font>입니다. 즉, 원래의 점 $f(x,y) = 0$ 을 $(p,q)$ 만큼 이동한다고 하면 $f(x-p, y-q) = 0$ 이 됩니다.

![[Pasted image 20240228142829.png]]

![[Pasted image 20240228142834.png]]

### 지수함수의 대칭 이동

지수함수 $y = a^x$ 의 대칭이동은 어떨까요? $x$ 축, $y$ 축 및 원점을 기준으로 대칭 이동했을 때를 알아봅시다. 지수함수 $y = a^x$ 그래프를 $x$ 축에 대칭 이동하면 $-y = a^x$ 가 되어 $y = -a^x$ 가 됩니다. 또 $y$ 축에 대칭 이동하면 $y = a^{-x}$ 가 되고, 원점에 대칭 이동하면 $-y = a^{-x}$ 가 됩니다.

![[Pasted image 20240228143104.png]]

![[Pasted image 20240228143108.png]]

즉, <font color="#ffff00">지수함수의 그래프는 x축과 만나지 않으며, 반드시 y축을 지나고 그 점은 1 또는 -1</font> 입니다.
