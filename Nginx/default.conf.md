- [[#Components of default.conf|Components of default.conf]]
	- [[#Components of default.conf#server|server]]
	- [[#Components of default.conf#location|location]]
	- [[#Components of default.conf#location /api|location /api]]
	- [[#Components of default.conf#upstream|upstream]]


`nginx.conf` 는 [[Nginx]] 서버의 전체 설정 파일이지만, `default.conf` 는 주로 <font color="#ffff00">기본적인 서버 블록설정을 정의하는 파일로 사용</font>됩니다. Nginx를 처음 설치하면 일반적으로 `default.conf` 파일이 생성되며, Nginx가 기본적으로 처리할 서버와 요청에 대한 설정이 들어있습니다.

이 파일은 기본적으로 <font color="#ffff00">/etc/nginx/conf.d/default.conf</font> 경로에 위치합니다.

## Components of default.conf

### server

- 서버 블록은 Nginx가 특정 도메인이나 IP 주소로 들어오는 요청을 처리하는 방식을 정의합니다.
- 여러 개의 서버 블록을 만들어서 도메인에 따라 서로 다른 애플리케이션이나 서비스를 제공할 수 있습니다.

```nginx
server {
    listen 80;  # 80번 포트에서 HTTP 요청을 수신
    server_name example.com;  # 서버가 처리할 도메인명

    location / {
        root /usr/share/nginx/html;  # 기본 웹 루트 디렉토리
        index index.html index.htm;  # 기본 제공할 파일
    }
}
```

> listen
- Nginx가 어떤 포트에서 요청을 받을지 지정합니다. 일반적으로 HTTP는 80번 포트, HTTPS는 443번 포트를 사용합니다.

> server_name
- 이 필드는 Nginx가 처리할 특정 도메인을 지정합니다. 위의 코드에서는 `example.com` 으로 들어오는 요청을 이 서버 블록이 처리하도록 지정합니다.
- 와일드카드(\*) 또는 여러 도메인을 지정하여 여러 호스트 이름에 대한 동일한 설정을 적용할 수 있습니다.

### location

- location 블록은 들어오는 요청의 URL 경로를 기반으로 특정 요청을 처리하는 방법을 정의합니다.
- 예를 들어 `/` 는 모든 경로를 의미하며, `/api` 로 설정하면 `/api` 경로로 들어오는 요청에 대한 처리 방법을 정의할 수 있습니다.

```nginx
location / {
    root /usr/share/nginx/html;
    index index.html;
}
```

> root
- `root` 는 Nginx가 정적 파일을 제공할 디렉토리 경로를 지정합니다. 위의 예에서는 `/usr/share/nginx/html` 디렉토리에서 정적 파일을 제공하도록 설정되어 있습니다.

> index
- `index` 는 클라이언트가 특정 경로를 지정하지 않고 요청할 때 제공할 기본 파일을 정의합니다. 예를 들어, `index index.html` 로 설정하면 사용자가 `example.com` 을 요청할 때 `index.html` 파일이 반환됩니다.

### location /api

- 특정 경로에서 API 서버 또는 다른 백엔드 서비스로 프록시 역할을 수행할 수 있습니다.

```nginx
location /api/ {
    proxy_pass http://backend_server;
	proxy_set_header Host $host;
	proxy_set_header X-Real-IP $remote_addr;
	proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	proxy_set_header X-Forwarded-Proto $scheme;
}
```

> proxy_pass http://backend_server

- \/api 로 들어온 요청을 `http://backend_server` 라는 주소로 전달합니다. 이때 `backend_server` 는 백엔드 서버의 이름 또는 IP 주소를 나타내며, 해당 서버가 Nginx와 <font color="#ffff00">동일한 네트워크 환경에 있다고 가정</font>합니다. 

> proxy_set_header Host $host;

- Nginx가 백엔드 서버로 요청을 전달할 때, 클라이언트의 원래 요청에 포함된 `Host` 헤더를 유지합니다.

> proxy_set_header X-Real-IP $remote_addr

- 클라이언트의 실제 IP 주소를 백엔드 서버로 전달합니다.

> proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for

- 요청이 여러 프록시를 거쳐 왔을 때 클라이언트의 원래 IP와 경로를 전달하는 데 사용되는 헤더입니다.

> proxy_set_header X-Forwarded-Proto $scheme

- 요청이 HTTP 또는 HTTPS로 들어왔는지에 대하 정보를 백엔드 서버에 전달합니다.

### upstream

`upstream` 블록은 Nginx 에서 여러 백엔드 서버로 트래픽을 분산(로드 밸런싱)할 때 사용하는 설정입니다. 이 블록은 Nginx가 리버스 프록시로 작동할 때 여러 서버를 그룹으로 묶어, 클라이언트의 요청을 그 중 하나로 전달할 수 있도록 합니다.

예를 들어, 여러 서버에서 동일한 애플리케이션을 실행하고 있다면, `upstream` 블록을 통해 트래픽을 균등하게 또는 특정 기준에 따라 분산시킬 수 있습니다. 이를 통해 Nginx는 부하를 분산하고 가용성을 높이는 로드 밸런서로도 기능할 수 있습니다.

```nginx
upstream backend_group {
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}
```

[[Docker]] 에서 Nginx의 `upstream` 설정을 사용하려면 Docker 컨테이너 내에서 Nginx가 다른 서비스 컨테이너로 트래픽을 라우팅할 수 있도록 설정해야 합니다. 보통 [[Docker compose]] 를 사용하여 여러 서비스를 정의하고, Nginx가 리버스 프록시 역할을 하도록 설정합니다. 이를 통해 Nginx는 여러 백엔드 애플리케이션 컨테이너로 요청을 분산시킬 수 있습니다.

`docker-compose.yaml` 파일을 사용하여 Nginx와 백엔드 서비스를 정의합니다.

```yaml
version: '3'
services:
  nginx:
    image: nginx:latest
    container_name: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - backend1
      - backend2

  backend1:
    image: my-backend-app:latest
    container_name: backend1
    expose:
      - "8080"
      
  backend2:
    image: my-backend-app:latest
    container_name: backend2
    expose:
      - "8080"

```

`nginx/default.conf` 파일을 통해 Nginx의 `upstream` 블록을 설정하고, 클라이언트로부터 들어오는 요청을 백엔드 컨테이너로 분산시킬 수 있습니다.

```nginx
upstream backend_group {
    server backend1:8080;  # backend1 컨테이너로의 연결
    server backend2:8080;  # backend2 컨테이너로의 연결
}

server {
    listen 80;

    location /api {
        proxy_pass http://backend_group;  # upstream으로 요청을 전달
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

```

