datasets 라이브러리는 다양한 소스로부터 데이터셋을 로드하고 사용하기 위한 도구를 제공합니다. 아래에서 설명된 다양한 방법을 사용하여 데이터셋을 로드할 수 있습니다.

1. <font color="#ffff00">The Hub without a datasey loading script</font>
- HuggingFace 모델 허브 (The Hub) 에서 미리 정의된 데이터셋을 직접 로드할 수 있습니다. 데이터셋 로딩 스크립트 없이 간단한 명령을 사용하여 데이터셋을 다운로드하고 사용할 수 있습니다.

2. <font color="#ffff00">Local loading script</font>
- 로컬 환경에서 데이터셋을 로드하기 위해 사용자 지정 로딩 스크립트를 작성하고 해당 스크립트를 사용하여 데이터셋을 로드할 수 있습니다. 이로써 사용자가 제어하는 데이터 로딩 로직을 적용할 수 있습니다.

3. <font color="#ffff00">Local files</font>
- 로컬 파일 시스템에 저장된 데이터 파일을 직접 로드할 수 있습니다. 데이터셋이 로컬 디스크에 있는 경우 해당 파일의 경로를 지정하여 데이터를 로드합니다.

4. <font color="#ffff00">Offline</font>
- 인터넷이 연결되지 않은 환경에서 데이터셋을 로드할 수 있습니다. 데이터셋을 로컬 파일 시스템에 미리 다운로드하고 오프라인 환경에서 사용할 수 있습니다.

5. <font color="#ffff00">A specific slice of a split</font>
- 데이터셋의 특정 부분만 필요한 경우, 데이터셋의 분할(split) 중 특정 부분만 선택하여 로드할 수 있습니다.

