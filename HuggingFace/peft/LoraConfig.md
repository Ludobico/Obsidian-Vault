> r -> int
- [[LoRA]] Attention의 차원입니다. LoRA는 low rank 근사를 통해 모델의 가중치 매트릭스를 압축하는 방식으로 동작합니다. 이 때 r은 근사 행렬의 rank를 결정합니다. 일반적으로 <font color="#ffff00">r이 작을수록 메모리 효율성은 높아지지만 성능이 저하될 수 있습니다</font>.

> target_modules -> Optional[Union[List[str], str]]
- LoRA를 적용할 모듈의 이름을 지정합니다. 문자열 하나를 전달하면 정규식 매칭을 수행하고, 문자열 리스트를 전달하면 정확히 매칭합니다. `all-linear` 를 지정하면 출력층을 제외한 모든 linear 모듈에 LoRA를 적용합니다. 지정하지 않으면 모델 아키텍처에 따라 자동으로 선택되지만, 아키텍처를 알 수 없는 경우 수동으로 지정해야 합니다.

> lora_alpha -> int
- LoRA 스케일링을 위한 alpha 파라미터입니다. 이 값에 따라 LoRA 가중치의 크기가 조절됩니다.

> lora_dropout -> float
- LoRA 레이어에 적용될 드롭아웃 확률입니다. <font color="#ffff00">드롭아웃은 과적합을 방지하고 성능을 높이기 위해 사용</font>됩니다. 

> bias -> str
- LoRA 에서의 바이어스 처리 유형을 지정합니다. 다음 세 가지 값 중 하나를 가집니다.
	-  `none` : 바이어스를 업데이트 하지 않습니다.
	- `all` : 모든 레이어의 바이어스를 업데이트합니다.
	- `lor_only` : LoRA 레이어의 바이어스만 업데이트합니다.
