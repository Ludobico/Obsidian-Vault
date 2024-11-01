- [[#Environment|Environment]]
- [[#Get key from Gabia|Get key from Gabia]]
- [[#Dockerfile setting|Dockerfile setting]]
- [[#Docker-compose setting|Docker-compose setting]]
- [[#Nginx setting|Nginx setting]]
	- [[#Nginx setting#default.conf|default.conf]]

## Environment
---
- web server : [[Nginx]]
- backend : [[Python]] FASTAPI
- CI/CD : [[Docker]]


## Get key from Gabia

![[스크린샷 2024-11-01 093313.jpg]]

가비아에서 SSL 인증을 받으면 \[my 가비아\] - \[서비스 관리\] 탭에서 SSL 보안서버 인증서를 확인할 수 있습니다.

인증서 정보탭을 보면 <font color="#ffff00">인증서 파일 요청</font> 을 클릭하여 파일을 다운로드 받습니다.

가비아에서 발급되는 인증서는 크게 4 종류로

- root_cert.crt (루트 인증서)
- chain_cert.crt (체인 인증서)
- cert.crt (도메인 인증서)
- .key

여기에서 **crt 파일을 조합하여 pem 키를 제작**할 수 있습니다.

```bash
cat [도메인 인증서] [체인 인증서] [루트인증서] > [원하는 이름.pem]
```

순서는 반드시 위의 순서대로 지켜야 합니다.

그렇게 발급받은 <font color="#ffff00">.pem</font> 과 <font color="#ffff00">.key</font> 를 Nginx 와 Docker 컨테이너에 적용합니다.

## Dockerfile setting

```dockerfile
FROM nginx
# 도커 컴포즈에서 도커파일 찾는 위치를 루트 디렉토리로 지정함
COPY path_to_nginx/default.conf /etc/nginx/conf.d/default.conf
COPY path_to_key/example.key /etc/nginx/ssl/example.key
COPY path_to_pem/example.pem /etc/nginx/ssl/example.pem
RUN apt-get update
RUN apt-get install vim -y
EXPOSE 8584
```


```dockerfile
COPY path_to_nginx/default.conf /etc/nginx/conf.d/default.conf
```

프로젝트 내의 Nginx의 설정을 담은 `default.conf` 파일을 도커 Nginx 컨테이너의 `etc/nginx/conf.d` 파일로 복사합니다.

```dockerfile
COPY path_to_key/example.key /etc/nginx/ssl/example.key
COPY path_to_pem/example.pem /etc/nginx/ssl/example.pem
```

**SSL 인증서 파일 복사**: 로컬의 SSL 키(`example.key`)와 인증서 파일(`example.pem`)을 각각 `/etc/nginx/ssl/` 폴더에 복사합니다. 이 파일들은 HTTPS 연결을 위한 인증서와 키로 사용됩니다.

```dockerfile
EXPOSE 8584
```

**포트 노출**: Nginx 서버가 외부와 통신할 수 있도록 **8584번 포트**를 엽니다. Docker Compose나 다른 도구에서 이 포트를 매핑하여 외부에서 접근할 수 있습니다.

만약 기본포트(80) 을 사용한다면 이 부분은 제외합니다.


## Docker-compose setting

```yaml
  backend_nginx:
    build:
      context: .
      dockerfile: ./path_to_nginx/dockerfile.nginx
    ports:
      - '8584:8584'
      - '443:443'
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf
    container_name: hops_nginx
    depends_on:
      - backend_blue
      - backend_green
    image: hops_image_nginx
    networks:
      - hops_network
```


**port** 같은경우 기본포트(80)를 사용한다면 다음과 같이 설정합니다.

```yaml
    ports:
      - '80:80'
      - '443:443'
```

**volumes** 의 경우 [[Dockerfile]] 에서 **COPY** 를 사용한다면 이 부분을 제외합니다.

```yaml
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf
```


## Nginx setting

### default.conf

```nginx
server {
    listen 443 ssl;
    listen 8584 ssl;
    server_name domain_name;

    # 보통 /etc/nginx/ssl 경로에 마운트
    ssl_certificate /etc/nginx/ssl/example.pem;
    ssl_certificate_key /etc/nginx/ssl/example.key;

    ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    location /api {
        proxy_pass http://upstream_name;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

만약 기본포트(80)를 사용한다면 SSL 443포트로 리다이렉션하는 기능을 추가로 작성합니다.

```nginx
    server {
        listen 80;
        server_name domain_name;

        return 301 https://$host$request_uri;
    }
```

