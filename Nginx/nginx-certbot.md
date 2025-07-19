
# init-letsencrypt.sh

## 스크립트 전체 구조

```bash
#!/bin/bash

if ! [ -x "$(command -v docker-compose)" ]; then
  echo 'Error: docker-compose is not installed.' >&2
  exit 1
fi

domains=(example.org www.example.org)
rsa_key_size=4096
data_path="./data/certbot"
email="" # Adding a valid address is strongly recommended
staging=0 # Set to 1 if you're testing your setup to avoid hitting request limits

if [ -d "$data_path" ]; then
  read -p "Existing data found for $domains. Continue and replace existing certificate? (y/N) " decision
  if [ "$decision" != "Y" ] && [ "$decision" != "y" ]; then
    exit
  fi
fi


if [ ! -e "$data_path/conf/options-ssl-nginx.conf" ] || [ ! -e "$data_path/conf/ssl-dhparams.pem" ]; then
  echo "### Downloading recommended TLS parameters ..."
  mkdir -p "$data_path/conf"
  curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$data_path/conf/options-ssl-nginx.conf"
  curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem > "$data_path/conf/ssl-dhparams.pem"
  echo
fi

echo "### Creating dummy certificate for $domains ..."
path="/etc/letsencrypt/live/$domains"
mkdir -p "$data_path/conf/live/$domains"
docker-compose run --rm --entrypoint "\
  openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1\
    -keyout '$path/privkey.pem' \
    -out '$path/fullchain.pem' \
    -subj '/CN=localhost'" certbot
echo


echo "### Starting nginx ..."
docker-compose up --force-recreate -d nginx
echo

echo "### Deleting dummy certificate for $domains ..."
docker-compose run --rm --entrypoint "\
  rm -Rf /etc/letsencrypt/live/$domains && \
  rm -Rf /etc/letsencrypt/archive/$domains && \
  rm -Rf /etc/letsencrypt/renewal/$domains.conf" certbot
echo


echo "### Requesting Let's Encrypt certificate for $domains ..."
#Join $domains to -d args
domain_args=""
for domain in "${domains[@]}"; do
  domain_args="$domain_args -d $domain"
done

# Select appropriate email arg
case "$email" in
  "") email_arg="--register-unsafely-without-email" ;;
  *) email_arg="--email $email" ;;
esac

# Enable staging mode if needed
if [ $staging != "0" ]; then staging_arg="--staging"; fi

docker-compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    $staging_arg \
    $email_arg \
    $domain_args \
    --rsa-key-size $rsa_key_size \
    --agree-tos \
    --force-renewal" certbot
echo

echo "### Reloading nginx ..."
docker-compose exec nginx nginx -s reload
```

```
1. docker-compose 설치 여부 확인
2. 변수 설정 (도메인, 이메일, 키 사이즈 등)
3. 이전 인증서 데이터가 있으면 사용자에게 덮어쓸지 확인
4. TLS 권장 설정 파일 다운로드
5. 더미 인증서 생성 (nginx가 SSL로 시작할 수 있도록)
6. nginx 컨테이너 실행
7. 더미 인증서 삭제
8. 진짜 인증서 발급 (Certbot 으로 Let's Encrypt 요청)
9. nginx 재시작 (새 인증서 반영)
```

### 각 단계 설명

```bash
if ! [ -x "$(command -v docker-compose)" ]; then
  echo 'Error: docker-compose is not installed.' >&2
  exit 1
fi
```

- [[Docker compose]] 명령어가 없는 경우 오류 메시지를 출력하고 종료

```bash
domains=(example.org www.example.org)
rsa_key_size=4096
data_path="./data/certbot"
email="" # Adding a valid address is strongly recommended
staging=0 # Set to 1 if you're testing your setup to avoid hitting request limits
```

- `domains`: 인증받을 도메인 목록
- `rsa_key_size`: SSL 키 사이즈 (4096 recommended)
- `data_path`: 인증서 파일을 저장할 경로
- `email`: Certbot 등록 이메일. 비워두면 이메일 없이 진행됨
- `staging`: 테스트용 인증서 발급 여부 (`1`이면 staging)

#### staging

Let's Encrypt는 전 세계에서 공짜로 SSL 인증서를 제공하는 서비스이기 때문에 **무분별한 요청을 막기 위해 발급 횟수 제한**이 있습니다.

- 같은 도메인에 대해 하루 5회까지만 발급 가능
- 실수로 스크립트를 여러 번 실행하면 제한에 걸려서 하루 동안 인증서 발급이 막힘

1. 테스트용 (staging = 1)

```bash
# 1. init-letsencrypt.sh 실행
./init-letsencrypt.sh

# 2. (필요 시) docker-compose up
docker-compose up -d
```

2. 실서비스 전환 (staging = 0)

```bash
# 1. 스크립트에서 staging=0 으로 수정
# 예: init-letsencrypt.sh 내부 변수만 변경

# 2. 다시 실행
./init-letsencrypt.sh
```


```bash
if [ -d "$data_path" ]; then
  read -p "Existing data found for $domains. Continue and replace existing certificate? (y/N) " decision
  if [ "$decision" != "Y" ] && [ "$decision" != "y" ]; then
    exit
  fi
fi
```

- 기존 인증서 디렉터리가 있을 경우, 덮어쓸지 여부 확인
- 사용자가 `y` 나 `Y` 를 입력하지 않으면 종료

```bash
if [ ! -e "$data_path/conf/options-ssl-nginx.conf" ] || [ ! -e "$data_path/conf/ssl-dhparams.pem" ]; then
  echo "### Downloading recommended TLS parameters ..."
  mkdir -p "$data_path/conf"
  curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$data_path/conf/options-ssl-nginx.conf"
  curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem > "$data_path/conf/ssl-dhparams.pem"
  echo
fi
```

- [[Nginx]] SSL 설정에 필요한 권장 TLS 파일을 certbot github 에서 다운로드
- 파일이 없다면 `conf` 디렉터리를 만들고 TLS 관련 파일들을 저장

```bash
echo "### Creating dummy certificate for $domains ..."
path="/etc/letsencrypt/live/$domains"
mkdir -p "$data_path/conf/live/$domains"
docker-compose run --rm --entrypoint "\
  openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1\
    -keyout '$path/privkey.pem' \
    -out '$path/fullchain.pem' \
    -subj '/CN=localhost'" certbot
echo
```

- nginx는 SSL 설정이 있으면 SSL 인증서가 존재해야 시작됨
- 그래서 임시(dummy) 인증서를 1일 짜리로 생성

```bash
echo "### Starting nginx ..."
docker-compose up --force-recreate -d nginx
echo
```

- nginx를 백그라운드에서 실행

```bash
echo "### Deleting dummy certificate for $domains ..."
docker-compose run --rm --entrypoint "\
  rm -Rf /etc/letsencrypt/live/$domains && \
  rm -Rf /etc/letsencrypt/archive/$domains && \
  rm -Rf /etc/letsencrypt/renewal/$domains.conf" certbot
echo
```

- 더미 인증서를 완전히 삭제하여 진짜 인증서로 대체할 준비

```bash
echo "### Requesting Let's Encrypt certificate for $domains ..."
#Join $domains to -d args
domain_args=""
for domain in "${domains[@]}"; do
  domain_args="$domain_args -d $domain"
done

# Select appropriate email arg
case "$email" in
  "") email_arg="--register-unsafely-without-email" ;;
  *) email_arg="--email $email" ;;
esac

# Enable staging mode if needed
if [ $staging != "0" ]; then staging_arg="--staging"; fi

docker-compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    $staging_arg \
    $email_arg \
    $domain_args \
    --rsa-key-size $rsa_key_size \
    --agree-tos \
    --force-renewal" certbot
echo
```

- 인증서 발급은 certbot이 `webroot` 방식을 사용하여 `.well-known/acme-challenge` 경로에 파일을 두고 검증을 받습니다.
- 여러 도메인을 한 번에 처리
- staging 여부, 이메일 여부에 따라 certbot 인자 구성

```bash
echo "### Reloading nginx ..."
docker-compose exec nginx nginx -s reload
```

- nginx 프로세스를 재시작하여 발급된 인증서를 반영합니다.

