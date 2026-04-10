출처 : https://xoft.tistory.com/109

Rotation matrix는 오일러 각(Euler Angle)을 기반으로 합니다. 물체의 회전을 나타내는 방법중에 하나입니다

2D에서 원점을 기준으로 각도 θ만큼 반시계 방향으로 회전시키는 행렬입니다.

$$
R = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

Rotation Matrix으로 point (x,y)를 회전하는 수식은 아래와 같습니다.

$$
\begin{bmatrix} x' \\ y'  \end{bmatrix} = R \cdot \begin{bmatrix} x \\ y  \end{bmatrix}
$$

$$
x' = x \cos \theta - y \sin \theta
$$
$$
y' = x \sin \theta + y \cos \theta
$$

점 (1,0)을 시작으로 θ가 10도씩 증가하는 예시입니다


![[img 1.gif]]