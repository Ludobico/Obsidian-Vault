
`YoutubeLoader` 는 **Youtube 동영상의 자막(transcripts)** 을 로드하기 위한 [[LangChain/LangChain|LangChain]] 의 [[document_loaders]] 입니다.

```python
class YoutubeLoader(
    video_id: str,
    add_video_info: bool = False,
    language: Union[str, Sequence[str]] = 'en',
    translation: Optional[str] = None,
    transcript_format: TranscriptFormat = TranscriptFormat.TEXT,
    continue_on_failure: bool = False,
    chunk_size_seconds: int = 120
)
```

## Parameters

> video_id -> str

- Youtube 동영상의 고유 ID 입니다.
- `https://www.youtube.com/watch?v=abc123`에서 `abc123`이 **video_id**입니다.

> add_video_info -> bool, Default False

- 동영상의 추가 정보(제목, 설명 등)을 자막과 함께 로드할지 여부를 설정합니다.

> language -> Union\[str, Sequence\[str]], Default 'en'

- 가져올 자막의 언어를 설정합니다.
- 기본값은 영어(en) 이며, 여러 언어를 리스트로 지정할 수 있습니다.
- \['en', 'es'] 로 설정하면 영어와 스페인어 자막을 가져옵니다.

> translation -> optional, str, Default : None

- 자막을 다른 언어로 번역할지 설정합니다.
- `translation='ko'` 로 설정하면 한국어로 번역된 자막을 반환합니다.

> chunk_size_seconds -> int, Default : 120

- 자막을 몇 초 단위로 나눌지 설정합니다.
- 기본값은 120초 단위로 자막을 나누어 반환합니다.

## Methods

> alazy_load() -> AsyncIterator\[Document]

- 자막 데이터를 비동기적으로 lazy load 합니다.

> aload() -> List\[Document]

- 자막 데이터를 비동기적으로 한 번에 모두 로드합니다.

> lazy_load() -> Iterator\[Document]

- 자막 데이터를 동기적으로 lazy load 합니다.

> load() -> List\[Document]

- 자막 데이터를 동기적으로 한 번에 모두 로드합니다.

> from_youtube_url() -> [[YoutubeLoader]]

- 주어진 YouTube URL을 기반으로 YoutubeLoader 인스턴스를 생성합니다.

- Parameters
	> youtube_url -> youtube 동영상의 url
	> \*\*kwargs -> youtubeLoader 에 전달할 추가 인자
	> 

```python
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=abc123", chunk_size_seconds=60)
documents = loader.load()
```

