
- [[#📋 사전 준비|📋 사전 준비]]
- [[#🚀 STEP 1: 폴더 생성 및 이동|🚀 STEP 1: 폴더 생성 및 이동]]
- [[#🚀 STEP 2: OpenClaw 소스 다운로드 및 이미지 빌드|🚀 STEP 2: OpenClaw 소스 다운로드 및 이미지 빌드]]
- [[#🚀 STEP 3: 환경 변수 파일 생성|🚀 STEP 3: 환경 변수 파일 생성]]
- [[#🚀 STEP 4: docker-compose.yml 보안 설정 패치 (필수!)|🚀 STEP 4: docker-compose.yml 보안 설정 패치 (필수!)]]
- [[#🚀 STEP 5: 온보딩 위자드 실행|🚀 STEP 5: 온보딩 위자드 실행]]
- [[#🚀 STEP 6: Gateway 토큰 확인 및 .env 업데이트|🚀 STEP 6: Gateway 토큰 확인 및 .env 업데이트]]
- [[#🚀 STEP 7: Gateway 시작|🚀 STEP 7: Gateway 시작]]
- [[#🚀 STEP 8: 웹 대시보드 접속 및 디바이스 페어링|🚀 STEP 8: 웹 대시보드 접속 및 디바이스 페어링]]
	- [[#🚀 STEP 8: 웹 대시보드 접속 및 디바이스 페어링#8-1. 브라우저에서 접속|8-1. 브라우저에서 접속]]
	- [[#🚀 STEP 8: 웹 대시보드 접속 및 디바이스 페어링#8-2. 토큰 입력|8-2. 토큰 입력]]
	- [[#🚀 STEP 8: 웹 대시보드 접속 및 디바이스 페어링#8-3. 디바이스 페어링 승인 (페어링 요청이 뜨면)|8-3. 디바이스 페어링 승인 (페어링 요청이 뜨면)]]
- [[#🚀 STEP 9: 설정 변경 (lan 모드로 전환) - 필수!|🚀 STEP 9: 설정 변경 (lan 모드로 전환) - 필수!]]
- [[#🚀 STEP 10: Discord 채널 연결|🚀 STEP 10: Discord 채널 연결]]
	- [[#🚀 STEP 10: Discord 채널 연결#10-1. Discord 봇 설정 (Discord Developer Portal)|10-1. Discord 봇 설정 (Discord Developer Portal)]]
	- [[#🚀 STEP 10: Discord 채널 연결#10-2. 봇을 서버에 초대|10-2. 봇을 서버에 초대]]
	- [[#🚀 STEP 10: Discord 채널 연결#10-3. 환경 변수 설정|10-3. 환경 변수 설정]]
	- [[#🚀 STEP 10: Discord 채널 연결#10-4. docker-compose.yml에 환경 변수 추가|10-4. docker-compose.yml에 환경 변수 추가]]
	- [[#🚀 STEP 10: Discord 채널 연결#10-5. openclaw.json에 Discord 채널 설정 추가|10-5. openclaw.json에 Discord 채널 설정 추가]]
	- [[#🚀 STEP 10: Discord 채널 연결#10-6. Gateway 재시작 및 Discord 활성화|10-6. Gateway 재시작 및 Discord 활성화]]
- [[#🚀 STEP 11: Discord DM 페어링 승인|🚀 STEP 11: Discord DM 페어링 승인]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#11-1. Discord에서 봇에게 DM 보내기|11-1. Discord에서 봇에게 DM 보내기]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#11-2. 페어링 승인 (exec 사용!)|11-2. 페어링 승인 (exec 사용!)]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#11-3. 연결 확인|11-3. 연결 확인]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#10-1. 소스 코드 다운로드 (ZIP 파일 받기)|10-1. 소스 코드 다운로드 (ZIP 파일 받기)]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#10-2. 필수 패키지 설치|10-2. 필수 패키지 설치]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#10-3. 환경 설정 (.env)|10-3. 환경 설정 (.env)]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#10-4. 서버 빌드 및 실행 (PM2 사용 - 필수)|10-4. 서버 빌드 및 실행 (PM2 사용 - 필수)]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#10-5. OpenClaw API 활성화 (필수 설정!)|10-5. OpenClaw API 활성화 (필수 설정!)]]
	- [[#🚀 STEP 11: Discord DM 페어링 승인#10-6. 카카오톡 연동 원리 (중요!)|10-6. 카카오톡 연동 원리 (중요!)]]
- [[#🚀 STEP 11: 외부 접속 주소 만들기 (선택)|🚀 STEP 11: 외부 접속 주소 만들기 (선택)]]
	- [[#🚀 STEP 11: 외부 접속 주소 만들기 (선택)#🎯 [방법 A] ngrok (가장 추천: 무료 고정 도메인 1개 제공)|🎯 [방법 A] ngrok (가장 추천: 무료 고정 도메인 1개 제공)]]
	- [[#🚀 STEP 11: 외부 접속 주소 만들기 (선택)#🌩️ [방법 B] Cloudflare Quick Tunnel (설치 간편, 임시 주소)|🌩️ [방법 B] Cloudflare Quick Tunnel (설치 간편, 임시 주소)]]
	- [[#🚀 STEP 11: 외부 접속 주소 만들기 (선택)#💎 [방법 C] Cloudflare Zero Trust (유료 도메인 필수, 전문가용)|💎 [방법 C] Cloudflare Zero Trust (유료 도메인 필수, 전문가용)]]
- [[#🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)|🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)]]
	- [[#🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)#12-1. 스킬 등록|12-1. 스킬 등록]]
	- [[#🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)#12-2. 시나리오 설정 (폴백 블록)|12-2. 시나리오 설정 (폴백 블록)]]
	- [[#🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)#12-3. 운영채널연결|12-3. 운영채널연결]]
	- [[#🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)#12-4. 콜백(Callback) 설정|12-4. 콜백(Callback) 설정]]
- [[#✅ 설치 완료!|✅ 설치 완료!]]
- [[#🔧 자주 사용하는 명령어|🔧 자주 사용하는 명령어]]
	- [[#🔧 자주 사용하는 명령어#💡 (꿀팁) 명령어 줄여서 쓰기 (Alias)|💡 (꿀팁) 명령어 줄여서 쓰기 (Alias)]]
- [[#⚠️ 문제 해결|⚠️ 문제 해결]]
	- [[#⚠️ 문제 해결#웹 대시보드 접속 안 될 때|웹 대시보드 접속 안 될 때]]
	- [[#⚠️ 문제 해결#Discord 봇이 오프라인일 때|Discord 봇이 오프라인일 때]]
	- [[#⚠️ 문제 해결#페어링이 안 될 때|페어링이 안 될 때]]
	- [[#⚠️ 문제 해결#CLI 연결 에러 (gateway closed 1006)|CLI 연결 에러 (gateway closed 1006)]]

# OpenClaw Docker 설치 - PowerShell 명령어 가이드

> **Windows + Docker Desktop 환경**  
> 처음부터 끝까지 순서대로 실행하세요.

---

## 📋 사전 준비

- ✅ Docker Desktop 설치 및 **실행 중**
- ✅ Discord 봇 토큰 준비 (Discord Developer Portal에서 생성)
- ✅ OpenAI API 키 준비 (또는 다른 AI 제공자)

---

## 🚀 STEP 1: 폴더 생성 및 이동

```powershell
# 작업 폴더 생성
mkdir d:\프로그램\openclaw
cd d:\프로그램\openclaw
```

> **주의:** 이 폴더는 비어 있어야 합니다. 기존 파일이 있다면 삭제하거나 다른 폴더를 사용하세요.

---

## 🚀 STEP 2: OpenClaw 소스 다운로드 및 이미지 빌드

**순서가 중요합니다!** Git Clone을 먼저 하고 나서 하위 폴더를 만들어야 에러가 나지 않습니다.

```powershell
# 1. Git Clone (소스 코드 다운로드)
git clone https://github.com/openclaw/openclaw.git .

# 2. 필수 하위 폴더 생성
mkdir .openclaw
mkdir workspace

# 3. Docker 이미지 빌드
# (⚠️ 중요: Docker Desktop이 켜져 있는지 꼭 확인하세요!)
docker build -t openclaw:local -f Dockerfile .
```

> ⏱️ 빌드에 5~10분 정도 소요됩니다. 빌드 에러가 나면 Docker Desktop 실행 여부를 확인하세요.

---

## 🚀 STEP 3: 환경 변수 파일 생성

`.env` 파일을 생성하고 아래 내용을 입력하세요:

```powershell
# .env 파일 생성 (메모장으로 편집)
notepad .env
```

**.env 파일 내용:**
```env
# 필수: 설정 및 작업공간 디렉토리
OPENCLAW_CONFIG_DIR=d:/프로그램/openclaw/.openclaw
OPENCLAW_WORKSPACE_DIR=d:/프로그램/openclaw/workspace

# Gateway 포트
OPENCLAW_GATEWAY_PORT=18789
OPENCLAW_BRIDGE_PORT=18790

# Docker 이미지
OPENCLAW_IMAGE=openclaw:local

# Gateway 토큰 (온보딩 후 자동 생성됨, 초기에는 비워두기)
OPENCLAW_GATEWAY_TOKEN=

# AI API 키 (사용할 제공자에 맞게 입력)
OPENAI_API_KEY=sk-proj-your-api-key-here

# Discord 봇 토큰
DISCORD_BOT_TOKEN=your-discord-bot-token-here
```

---

## 🚀 STEP 4: docker-compose.yml 보안 설정 패치 (필수!)

최신 버전의 OpenClaw에서 발생하는 **웹챗 페어링 문제**와 **IP 주소 오류(172.x.x.x)**를 해결하기 위해 설정을 변경합니다. 이 설정은 외부 접속을 차단하고 **내 컴퓨터(localhost)**에서만 접속하도록 강제하여 보안을 강화합니다.

```powershell
notepad docker-compose.yml
```

파일을 열고 `ports:` 부분을 찾아 아래와 같이 수정하세요 (약 20~30번째 줄):

**변경 전:**
```yaml
    ports:
      - "${OPENCLAW_GATEWAY_PORT}:${OPENCLAW_GATEWAY_PORT}"
```

**변경 후:** (앞에 `127.0.0.1:` 을 추가)
```yaml
    ports:
      - "127.0.0.1:${OPENCLAW_GATEWAY_PORT}:${OPENCLAW_GATEWAY_PORT}"
```

> **💡 왜 이렇게 하나요?**  
> 도커의 내부 IP가 아닌 **로컬호스트(127.0.0.1)**로만 접속을 허용하여 브라우저 보안 경고(CORS)를 해결하고, 웹챗 페어링이 실패하는 문제를 완벽하게 고칩니다.

---

## 🚀 STEP 5: 온보딩 위자드 실행

```powershell
docker compose run --rm openclaw-cli onboard
```

**온보딩 과정:**
1. Continue? → **Yes**
2. Onboarding mode → **QuickStart** (권장)
3. AI Provider → **OpenAI** (또는 원하는 제공자)
4. API Key → (입력)
5. Model → **gpt-5-mini** (또는 원하는 모델)

> 온보딩 완료 후 `.openclaw/openclaw.json` 파일이 생성됩니다.

---

## 🚀 STEP 6: Gateway 토큰 확인 및 .env 업데이트

```powershell
# Gateway URL 및 토큰 확인
docker compose run --rm openclaw-cli dashboard --no-open
```

출력된 **토큰**을 `.env` 파일의 `OPENCLAW_GATEWAY_TOKEN=` 에 붙여넣으세요.

---

## 🚀 STEP 7: Gateway 시작

```powershell
docker compose up -d openclaw-gateway
```

**로그 확인 (선택사항):**
```powershell
docker compose logs -f openclaw-gateway
```

> `Ctrl + C`로 로그 보기 종료

---

## 🚀 STEP 8: 웹 대시보드 접속 및 디바이스 페어링

### 8-1. 브라우저에서 접속
```
http://localhost:18789
```

### 8-2. 토큰 입력
- Settings (⚙️) 클릭
- Gateway Token 필드에 토큰 붙여넣기
- Save

### 8-3. 디바이스 페어링 승인 (페어링 요청이 뜨면)

> ⚠️ **중요**: Docker 환경에서는 `docker compose run`이 아닌 `docker compose exec`를 사용해야 합니다!

```powershell
# 대기 중인 디바이스 목록 확인 (exec 사용!)
docker compose exec openclaw-gateway node dist/index.js devices list	

# 디바이스 승인 (requestId를 위 명령어 결과에서 확인)
docker compose exec openclaw-gateway node dist/index.js devices approve <requestId>
```

**예시:**
```powershell
# Pending 목록에서 Request ID 확인 후:
docker compose exec openclaw-gateway node dist/index.js devices approve xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

## 🚀 STEP 9: 설정 변경 (lan 모드로 전환) - 필수!

페어링이 성공적으로 완료되었다면, 이제 CLI 명령어를 편하게 사용하고 외부 봇(카카오톡 등)과 원활하게 통신하기 위해 설정을 변경합니다.

1. `.openclaw/openclaw.json` 원본 파일을 메모장으로 엽니다.
2. `gateway` 섹션의 `"bind": "loopback"` 부분을 찾아서 **`"bind": "lan"`**으로 변경하고 저장합니다. (약 10~20번째 줄)

```json
"gateway": {
  "mode": "local",
  "bind": "lan",  // ← loopback에서 lan으로 변경!
  ...
}
```

3. **Docker 재시작:**
   ```powershell
   docker compose restart openclaw-gateway
   ```

> **💡 왜 바꾸나요?**  
> 페어링은 보안 문제로 `loopback` 상태에서 해야 잘 되지만, 이후 다른 명령어(`openclaw-cli`)나 카카오톡 연동은 `lan` 모드에서 훨씬 간편하게 작동하기 때문입니다. 이미 페어링된 정보는 사라지지 않으니 안심하세요!

---

## 🚀 STEP 10: Discord 채널 연결

### 10-1. Discord 봇 설정 (Discord Developer Portal)
1. https://discord.com/developers/applications 접속
2. **New Application** → 봇 이름 입력 → Create
3. **Bot** 메뉴 → **Reset Token** → 토큰 복사 (안전한 곳에 저장!)
4. **Privileged Gateway Intents** 3개 모두 켜기:
   - ✅ PRESENCE INTENT
   - ✅ SERVER MEMBERS INTENT
   - ✅ MESSAGE CONTENT INTENT
5. **Save Changes**

### 10-2. 봇을 서버에 초대
1. 좌측 메뉴에서 **OAuth2** 클릭
2. 페이지 하단 **"OAuth2 URL Generator"** 섹션으로 스크롤
3. **Scopes**에서 ✅ `bot` 체크
4. **Bot Permissions**에서:
   - ✅ `Send Messages`
   - ✅ `Read Message History`
   - ✅ `View Channels`
5. 하단 **"Generated URL"** 복사 → 브라우저에서 열기 → 서버 선택 → **Authorize**

### 10-3. 환경 변수 설정

**`.env` 파일에 Discord 토큰 추가:**
```powershell
notepad .env
```

```env
# Discord 봇 토큰 추가
DISCORD_BOT_TOKEN=your-discord-bot-token-here
```

### 10-4. docker-compose.yml에 환경 변수 추가

> ⚠️ **중요**: `docker-compose.yml`의 `openclaw-gateway` 서비스 `environment` 섹션에 다음 줄이 있는지 확인하세요:

```yaml
services:
  openclaw-gateway:
    environment:
      # ... 기존 환경 변수들 ...
      DISCORD_BOT_TOKEN: ${DISCORD_BOT_TOKEN}  # ← 이 줄 추가!
```

### 10-5. openclaw.json에 Discord 채널 설정 추가

`.openclaw/openclaw.json` 파일을 열고, **`gateway` 블록이 끝난 다음**에 `channels` 섹션을 추가하세요.
**⚠️ 주의: `gateway` 설정 안쪽(`{ ... }`)에 넣으면 에러가 납니다!**

**올바른 위치 예시:**
```json
  "gateway": {
      ...
  },  // ← 여기서 gateway 끝남

  // ▼ 여기에 추가하세요!
  "channels": {
    "discord": {
      "enabled": true,
      "groupPolicy": "allowlist"
    }
  },

  "plugins": { ... }
```

### 10-6. Gateway 재시작 및 Discord 활성화

```powershell
# Gateway 완전 재시작 (환경 변수 적용)
docker compose down
docker compose up -d openclaw-gateway

# Discord 자동 활성화
docker compose exec openclaw-gateway node dist/index.js doctor --fix
```

> 💡 `doctor --fix`를 실행하면 "Discord configured, not enabled yet" 메시지가 사라지고 Discord가 활성화됩니다.

---

## 🚀 STEP 11: Discord DM 페어링 승인

### 11-1. Discord에서 봇에게 DM 보내기
봇에게 "안녕" 이라고 메시지를 보내면 **페어링 코드**가 옵니다.

예시 응답:
```
OpenClaw: access not configured.
Your Discord user id: 123456789012345678
Pairing code: ABC12XYZ

Ask the bot owner to approve with:
openclaw pairing approve discord <code>
```

### 11-2. 페어링 승인 (exec 사용!)

> ⚠️ **중요**: `docker compose run`이 아닌 `docker compose exec`를 사용하세요!

```powershell
# 페어링 코드로 승인 (ABC12XYZ를 실제 페어링 코드로 교체)
docker compose exec openclaw-gateway node dist/index.js pairing approve discord ABC12XYZ

# 또는 모든 대기 중인 페어링 승인
docker compose exec openclaw-gateway node dist/index.js pairing approve-all discord
```

### 11-3. 연결 확인

페어링이 완료되면:
1. Discord 봇이 🟢 **온라인** 상태가 됩니다
2. 봇에게 메시지를 보내면 AI가 응답합니다
3. **웹챗(Control UI)**에서도 Discord 대화 내용을 볼 수 있습니다 (정상 동작)

---

# 🚀 STEP 10: 카카오톡 브릿지 서버 설치 (통역사 서버)
오픈클로(OpenClaw)와 카카오톡을 연결해주는 **브릿지 서버**를 설치합니다. 이 서버는 카카오톡의 신호를 OpenClaw가 이해할 수 있게 번역해줍니다.
### 10-1. 소스 코드 다운로드 (ZIP 파일 받기)

카카오톡 봇 소스 코드를 다운로드해서 **워크스페이스 폴더** 안에 넣습니다.

1. **[GitHub 저장소 받기](https://github.com/tornado1014/clawdbot-kakaotalk/archive/refs/heads/main.zip)** 클릭하여 `ZIP` 파일을 다운로드합니다.
2. 압축을 풀면 나오는 폴더 이름을 **`clawdbot-kakaotalk`** 로 바꿉니다.
3. 이 폴더를 **`d:\프로그램\openclaw\workspace`** 안으로 이동시킵니다.

**결과 확인:**
```powershell
# 폴더 구조가 이렇게 되어야 합니다:
# d:\프로그램\openclaw\workspace\clawdbot-kakaotalk\package.json
```

**터미널에서 이동:**
```powershell
# 워크스페이스 안의 봇 폴더로 이동
cd d:\프로그램\openclaw\workspace\clawdbot-kakaotalk
```
### 10-2. 필수 패키지 설치
서버 실행에 필요한 라이브러리를 설치합니다. (Node.js 18 이상 권장)
```powershell
npm install
```
### 10-3. 환경 설정 (.env)
설정 파일 예시를 복사하고 내 환경에 맞게 수정합니다.
```powershell
# .env 파일 생성
Copy-Item .env.example .env
# 메모장으로 편집
notepad .env
```
**수정할 내용:**
- `CLAWDBOT_GATEWAY_URL`: `http://localhost:18789` (내 OpenClaw 주소)
- `CLAWDBOT_GATEWAY_TOKEN`: **STEP 5**에서 확인한 Gateway 토큰값
- `PAIRING_CODE`: 카카오톡 인증용 비밀번호 (예: `1234`)
- `CLAWDBOT_MODEL`: 사용할 모델명 (예: `google-antigravity/gemini-3-flash`)
### 10-4. 서버 빌드 및 실행 (PM2 사용 - 필수)

PM2를 사용하면 서버가 꺼져도 자동으로 다시 켜주니까 훨씬 안정적입니다. 🌙

1. **프로젝트 빌드 (필수)**
   PM2는 빌드된 자바스크립트 파일(`dist/index.js`)을 실행하므로 먼저 빌드를 해야 합니다.

   ```powershell
   cd d:\프로그램\openclaw\workspace\clawdbot-kakaotalk
   npm run build
   ```

2. **PM2로 실행하기**
   저장소에 이미 설정 파일(`ecosystem.config.js`)이 준비되어 있어 간단합니다.

   ```powershell
   # 카카오톡 브릿지만 실행
   pm2 start ecosystem.config.js --only clawdbot-kakaotalk
   ```
   > (참고: 설정 파일에 gateway도 있지만, OpenClaw는 이미 Docker로 돌고 있으니 브릿지만 켜면 됩니다!)

3. **관리 명령어**
   ```powershell
   # 상태 확인
   pm2 status
   
   # 로그 보기
   pm2 logs clawdbot-kakaotalk
   
   # 서버 끄기
   pm2 stop clawdbot-kakaotalk
   
   # 재시작
   pm2 restart clawdbot-kakaotalk
   ```

> **💡 주의사항:** PM2로 실행한 후에는 기존에 켜뒀던 `npm run dev` 창은 닫으세요. 두 개가 동시에 켜지면 에러가 납니다.

### 10-5. OpenClaw API 활성화 (필수 설정!)

카카오톡 브릿지 서버는 OpenClaw와 대화할 때 'OpenAI 호환 API' 방식을 사용합니다. 그런데 OpenClaw는 보안상 이 기능을 기본적으로 꺼두는 경우가 많아 **반드시 활성화(true)** 해줘야 합니다.

**방법 1: 명령어(CLI)로 켜기 (가장 추천)**
터미널에서 아래 명령어를 입력하면 즉시 설정이 반영됩니다.

```powershell
docker compose exec openclaw-gateway node dist/index.js config patch '{"gateway":{"http":{"endpoints":{"chatCompletions":{"enabled":true}}}}}'
```

**방법 2: 설정 파일 직접 수정**
`.openclaw/openclaw.json` 파일을 열고 `gateway` 섹션을 찾아 아래 내용을 추가/수정하세요.

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": {
          "enabled": true
        }
      }
    }
    // ... 기존 설정들 ...
  }
}
```

**적용:** 설정을 바꾼 후에는 OpenClaw를 재시작해야 합니다.
```powershell
docker compose restart openclaw-gateway
```

> **💡 TIP:** "AI 처리 중 오류가 발생했습니다" 메시지가 계속 뜨면, 1) 위 API 설정이 켜져 있는지, 2) `.env` 파일의 `CLAWDBOT_GATEWAY_TOKEN`이 정확한지 꼭 확인하세요! 🌙

### 10-6. 카카오톡 연동 원리 (중요!)
카카오톡 봇이 정상 작동하려면 **두 개의 서버가 모두 켜져 있어야 합니다.**
1. **OpenClaw (브레인):** Docker로 실행 중 (`http://localhost:18789`)
2. **브릿지 서버 (통역사):** PM2로 실행 중 (`http://localhost:3000`)
> 카카오톡에서 보낸 메시지는 **Cloudflare Tunnel (외부 주소) → 브릿지 서버 → OpenClaw** 순서로 전달됩니다. 컴퓨터를 끄면 봇도 멈추니 주의하세요!
---
## 🚀 STEP 11: 외부 접속 주소 만들기 (선택)

카카오톡 서버가 내 컴퓨터(브릿지 서버)에 접속할 수 있도록 외부 주소를 만들어야 합니다. 아래 3가지 방법 중 하나를 선택하세요.

### 🎯 [방법 A] ngrok (가장 추천: 무료 고정 도메인 1개 제공)
전 세계에서 가장 유명한 도구입니다. **최신 정책 변경으로 무료 사용자도 고정 도메인 1개를 평생 무료로 쓸 수 있습니다.** (가장 안정적!)

1. **가입 & 설치:** [ngrok.com](https://ngrok.com) 가입 → 대시보드에서 윈도우용 다운로드
2. **고정 도메인 만들기:**
   - 대시보드 좌측 메뉴 **Cloud Edge > Domains** 클릭
   - **+ Create Domain** 클릭 (예: `my-moonbot.ngrok-free.app` 같은 주소가 생성됨)
3. **인증 & 실행:**
   - 대시보드 첫 화면의 `Connect your account`에 있는 인증 명령어 복사해서 실행 (`ngrok config add-authtoken ...`)
   - 아래 명령어로 실행 (내 도메인 주소 넣기):
   ```powershell
   ngrok http --domain=내-도메인-주소.ngrok-free.app 3000
   ```
- **주소:** `https://내-도메인-주소.ngrok-free.app` (영원히 안 바뀜!)
- **장점:** 속도 빠름, 웹 관리 도구 제공, 가장 안정적.

---

### 🌩️ [방법 B] Cloudflare Quick Tunnel (설치 간편, 임시 주소)
회원가입 없이 바로 쓸 수 있지만, 켤 때마다 주소가 바뀝니다. 잠깐 테스트할 때만 쓰세요.

1. **설치:** `winget install Cloudflare.cloudflared`
2. **실행:**
   ```powershell
   cloudflared tunnel --url http://localhost:3000
   ```
- **주소:** 실행 시 나오는 `https://....trycloudflare.com` 복사
- **단점:** 터미널을 껐다 켜면 주소가 바뀌어서 카카오톡 설정도 매번 바꿔야 합니다.

---

### 💎 [방법 C] Cloudflare Zero Trust (유료 도메인 필수, 전문가용)
가장 안정적이고 끊김이 없지만, **내 개인 도메인(.com 등)**이 있어야만 쓸 수 있습니다.

1. **Cloudflare Zero Trust** 접속 → **Networks > Tunnels** → **Create a tunnel**
2. **Connector** 생성 명령어 복사 → PowerShell에 붙여넣기 (서비스로 자동 설치됨)
3. **Public Hostnames** 추가:
   - **Service Type:** `HTTP`
   - **URL:** `localhost:3000`
- **주소:** `https://내가설정한도메인.com`
- **장점:** 터미널을 닫아도 백그라운드에서 계속 실행됩니다.

---

> 💡 **최종 선택 가이드:**
> - **가장 추천 (무료/고정주소)** 👉 **[방법 A] ngrok**
> - **잠깐만 테스트할 거임** 👉 **[방법 B] Cloudflare Quick Tunnel**
> - **내 도메인 있고 24시간 돌릴 거임** 👉 **[방법 C] Zero Trust**

생성된 주소를 복사해두세요. (예: `https://my-moonbot.ngrok-free.app`)

---

## 🚀 STEP 12: 카카오톡 챗봇 관리자 센터 설정 (필수)

방금 만든 외부 주소를 카카오톡 챗봇에 연결하고, 시나리오를 설정하는 단계입니다.
[카카오톡 챗봇 관리자 센터](https://chatbot.kakao.com)에 로그인해서 내 챗봇을 선택하세요.

### 12-1. 스킬 등록
1. 왼쪽 메뉴에서 **스킬(Skill)** > **스킬 목록**을 클릭합니다.
2. **+ 생성** 버튼을 누릅니다.
3. 정보를 입력합니다:
   - **스킬 이름:** (예: `문봇3` 등 자유롭게)
   - **URL:** 아까 복사한 주소 뒤에 `/skill`을 붙입니다.
     - (예: `https://my-moonbot.ngrok-free.app/skill`)
   - **기본 스킬로 설정:** ✅ **체크해주세요.**
4. **저장**을 누릅니다.

### 12-2. 시나리오 설정 (폴백 블록)
사용자가 챗봇이 모르는 말을 했을 때 AI가 대답하게 만드는 설정입니다.

1. 왼쪽 메뉴에서 **시나리오(Scenario)**를 클릭합니다.
2. **기본 시나리오** 목록에서 **폴백 블록(Fallback Block)**을 선택합니다.
3. **파라미터 설정** 섹션에서:
   - 오른쪽 드롭다운 메뉴를 눌러 방금 만든 스킬(예: `문봇3`)을 선택합니다.
4. **봇 응답** 섹션에서:
   - **`<> 스킬데이터 사용`** 버튼을 클릭합니다.
   - (버튼이 활성화되어 파란색 테두리 등이 생겼는지 확인하세요)
5. **저장**을 누릅니다.

### 12-3. 운영채널연결
1. 상단의 **파트너**로 이동해서 채널 > 챗봇연결을 눌러서 내 카카오톡 채널과 챗봇을 **연결(ON)** 합니다.
2. 다시 상단의 **비즈도구>챗봇**을 누르고 아까 만든 봇을 누르고 **배포(Deploy)**를 클릭합니다.
3. **배포** 버튼을 눌러서 현재 버전을 배포합니다. 

### 12-4. 콜백(Callback) 설정
AI가 생각하는 동안 "잠시만 기다려주세요" 같은 반응을 하도록 설정합니다.

1. 폴백 블록 화면 오른쪽 위의 **점 3개(...)** 버튼을 클릭합니다.
2. **Callback 설정**을 누릅니다.
3. 설정을 켜고 저장합니다.
4. **배포** 버튼을 눌러서 현재 버전을 배포합니다. 

이제 카카오톡에서 봇에게 말을 걸면 AI가 대답합니다! 🎉

---
## ✅ 설치 완료!

이제 Discord와 카카오톡에서 봇과 대화할 수 있습니다! 🎉

---

## 🔧 자주 사용하는 명령어

**기본적으로 아래 명령어들을 복사해서 사용하세요. (가장 안정적)**

```powershell
# Gateway 시작
docker compose up -d openclaw-gateway

# Gateway 중지
docker compose down

# Gateway 재시작
docker compose restart openclaw-gateway

# Gateway 로그 보기 (실시간)
docker compose logs -f openclaw-gateway

# 상태 확인
docker compose ps

# 대기 중인 페어링 목록
docker compose exec openclaw-gateway node dist/index.js pairing list discord

# 디바이스 목록
docker compose exec openclaw-gateway node dist/index.js devices list

# OpenClaw 업데이트
docker compose exec openclaw-gateway node dist/index.js update
```

---

### 💡 (꿀팁) 명령어 줄여서 쓰기 (Alias)

매번 `docker compose exec...` 치기 귀찮다면, **터미널을 켤 때마다** 아래 명령어를 한 번만 입력하세요.
그러면 해당 창이 닫힐 때까지는 `openclaw` 명령어만으로 도커 내부 기능을 바로 쓸 수 있습니다!

```powershell
function openclaw { docker compose exec openclaw-gateway node dist/index.js $args }
```

**사용 예시:**
- `openclaw onboard`
- `openclaw devices list`
- `openclaw update`

---

## ⚠️ 문제 해결

### 웹 대시보드 접속 안 될 때
1. Gateway가 실행 중인지 확인: `docker compose ps`
2. 포트 확인: `docker compose logs openclaw-gateway | Select-String "listening"`
3. 방화벽에서 18789 포트 허용

### Discord 봇이 오프라인일 때
1. 토큰이 올바른지 확인
2. Gateway 재시작: `docker compose restart openclaw-gateway`
3. Privileged Intents가 켜져 있는지 확인

### 페어링이 안 될 때
```powershell
# 모든 페어링 강제 승인
docker compose exec openclaw-gateway node dist/index.js pairing approve-all discord
```

### CLI 연결 에러 (gateway closed 1006)

**에러 메시지:**
```
[openclaw] CLI failed: Error: gateway closed (1006 abnormal closure)
Gateway target: ws://127.0.0.1:18789
```

**원인:** `docker compose run`은 별도 컨테이너를 생성하여 Gateway에 연결할 수 없음

**해결책:** `docker compose exec` 사용 (이미 실행 중인 Gateway 컨테이너 내부에서 명령 실행)

```powershell
# ❌ 안 됨:
docker compose run --rm openclaw-cli devices list

# ✅ 됨:
docker compose exec openclaw-gateway node dist/index.js devices list
```

---

**📅 마지막 업데이트: 2026-02-09**
