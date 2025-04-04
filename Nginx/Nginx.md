![[Pasted image 20241024142811.png]]

Nginx는 오픈 소스 웹 서버이자 리버스 프록시 서버, 로드 밸런서, HTTP 캐시 역할을 하는 소프트웨어입니다.

### 주요 기능:

1. **웹 서버**: Nginx는 정적 콘텐츠(HTML, CSS, 이미지 등)를 빠르게 처리하는 데 매우 효율적입니다. 이 기능을 통해 사용자는 정적인 웹 페이지를 제공할 수 있습니다.
    
2. **리버스 프록시 서버**: 클라이언트 요청을 받아 백엔드 서버에 전달하고, 백엔드 서버의 응답을 클라이언트에 반환하는 역할을 합니다. 이 방식은 보안 및 성능 향상에 기여하며, 백엔드 서버의 IP를 숨기고 여러 서버로 요청을 분산시킬 수 있습니다.
    
3. **로드 밸런서**: Nginx는 여러 서버에 트래픽을 분산하여 서버의 과부하를 방지하고 성능을 최적화합니다. 로드 밸런싱을 통해 서버 간의 부하를 고르게 분산하고, 장애 발생 시에도 가용성을 유지할 수 있습니다.
    
4. **HTTP 캐싱**: Nginx는 프록시 캐시 기능을 제공하여, 서버 응답을 캐싱하고 이후 요청에서 캐시된 콘텐츠를 제공함으로써 서버 로드를 줄이고 성능을 향상시킵니다.
    
5. **TLS/SSL 지원**: 보안을 위해 HTTPS 프로토콜을 지원하며 SSL 인증서를 사용하여 암호화된 연결을 제공합니다.

