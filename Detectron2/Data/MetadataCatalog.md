<font color="#ffc000">MetadataCatalog</font> 는 [[Detectron2]] 프레임워크에서 사용되는 구성 요소로, 주어진 <font color="#ffff00">데이터셋에 대한 메타데이터에 액세스할 수 있는 전역 dictionary</font>입니다. 이 사전은 데이터셋의 이름에 해당하는 문자열을 해당 데이터셋의 메타데이터에 매핑합니다.

메타데이터는 한 번 생성되면 프로그램 실행 중에 유지되는 싱글톤 패턴을 따르며, 이후에 동일한 데이터셋에 대한 호출에서 동일한 메타데이터 인스턴스를 반환합니다. 이는<font color="#ffff00"> 프로그램 실행 중에 데이터셋에 대한 메타정보를 공유하고 재사용</font>하는 데 유용합니다.

### `MetadataCatalog.get`(_name_)
> name -> str
- 데이터셋의 이름, 예를 들어 "coco_2014_train"과 같은 문자열로 데이터셋을 식별합니다.

