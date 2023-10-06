COCO(Common Object in Context)는 객체 탐지, 분할 및 키포인트 검출을 위한 일련의 데이터 형식과 국제적으로 인정받은 대규모 데이터셋입니다. COCO 데이터셋은 객체 인식 및 인스턴스 분할 작업에 많이 사용됩니다.

COCO 데이터셋은 다양한 객체 인스턴스를 포함하고 있으며, 객체의 경계 상자와 객체의 픽셀 단위 세그멘테이션 정보를 제공합니다. 이를 통해 객체 인식, 객체 분할, 객체 검출 등의 작업에 활용할 수 있습니다.

COCO 데이터셋은 컴퓨터 비전 연구 및 개발을 위해 널리 사용되며, 객체 감지와 분할 알고리즘의 평가 및 벤치마킹에도 사용됩니다.

## <font color="#ffc000">Structure</font>
```json
{ "info": {...}, 
   "licenses": [...], 
   "images": [...], 
   "annotations": [...], 
   "categories": [...], <-- Not in Captions annotations 
   "segment_info": [...] <-- Only in Panoptic annotations }
```
original COCO Format의 필수인 항목은 <font color="#00b050">녹색</font>으로 표시합니다.

### <font color="#ffc000">info</font>
dict 구조로 전체적인 데이터의 정보들이 나열되어 있습니다.

- description
해당 annotation 파일에 대한 설명이 담겨 있습니다.

- contributor
데이터 및 라벨을 구축하는데 참여한 인원 혹은 집단의 정보가 담겨있습니다.

- url
데이터 및 라벨 제공자의 url 정보가 담겨있습니다.

- data_created
구축된 날짜 정보가 담겨있습니다. export 가 되는 시점의 ISO Date, Time format으로 들어갑니다.

- year
데이터셋이 생성 또는 업데이트된 연도를 나타냅니다.

- version
데이터셋의 버전을 나타냅니다.

### <font color="#ffc000">licenses</font>
list 구조로 지적 재산권 정보가 담겨있습니다.

- id
라이센스의 고유 식별자를 나타내는 정수형 필드입니다. 이 식별자는 라이센스를 구분하는데 사용되며 각 라이센스는 고유한 id 값을 가지고 있어야합니다.

- name
라이센스의 이름을 나타내는 문자열 필드입니다. 이 필드에는 라이센스의 이름이나 제목을 기재합니다. 예를 들어 *Creative Commons Attribution 4.0* 과 같은 라이센스의 이름을 포함합니다.

- url
라이센스에 대한 url을 나타내는 문자열 필드입니다. 이 필드에는 라이센스의 대한 상세한 정보를 제공하는 웹 페이지의 주소를 기재합니다.

### <font color="#ffc000">categories</font>
list 구조로 annotation에 들어있는 object의 class 정보들이 담겨있습니다.

- <font color="#00b050">id</font>
카테고리의 id로, 정수값으로 1부터 순차적으로 증가합니다.

- <font color="#00b050">name</font>
카테고리의 이름입니다.

- <font color="#00b050">supercategory</font>
상위 카테고리의 이름입니다.

### <font color="#ffc000">images</font>
list 구조로 raw data(이미지)에 대한 정보가 담겨있습니다.

- <font color="#00b050">id</font>
이미지의 id로, 정수값으로 1부터 순차적으로 증가합니다.

- <font color="#00b050">file_name</font>
이미지의 파일명이 담겨있습니다.

- <font color="#00b050">width</font>
이미지의 가로 픽셀 길이가 담겨있습니다.

- <font color="#00b050">height</font>
이미지의 높이 픽셀 길이가 담겨있습니다.

- dataset
이미지 데이터셋의 이름입니다.

- license
이미지에 적용된 라이센스의 식별자를 나타냅니다.

- flickr_url
이미지가 Flickr에서 제공된 경우 해당 이미지의 Flickr url 을 나타냅니다.

- coco_url
이미지가 coco 데이터셋 웹사이트에서 제공되는 경우 해당 이미지의 url을 나타냅니다.

- date_aptured
이미지가 캡처된 날짜/시간 정보를 나타냅ㄴ디ㅏ.

### <font color="#ffc000">annotations</font>
list 구조로 실질적인 라벨 정보가 담겨있습니다.

- <font color="#00b050">id</font>
annotation의 id로, 정수값으로 1부터 순차적으로 증가합니다.

- <font color="#00b050">image_id</font>
annotation에 해당하는 이미지의 id입니다.

- <font color="#00b050">category_id</font>
annotation에 해당하는 카테고리 id 입니다.

- segmentation
객체의 분할 정보를 나타냅니다. object detection에서 bbox만 검출하는 경우는 필수가 아니지만, <font color="#ffff00">segmentation 작업을 진행할 경우 필수</font>로 기입합니다.
이 필드는 V2 RLE(Run-Length-Encoding) 형식 또는 V1 Polygon 형식으로 표현합니다.

- <font color="#00b050">area</font>
객체의 면적을 나타내는 값입니다. 일반적으로 픽셀 단위로 표현됩니다.

- <font color="#00b050">bbox</font>
segmentation을 감싼 bounding box의 좌표정보가 들어있습니다.

- <font color="#00b050">iscrowd</font>
객체가 다수의 인스턴스로 구성되어 있는지 여부를 나타냅니다. 0은 개별 객체, 1은 다수의 객체를 나타냅니다.