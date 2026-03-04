## OpenClaw 깃허브 소스코드 클론

가장 먼저 OpenClaw의 공식 소스코드를 내 PC로 다운로드(클론)하고, 해당 폴더로 이동합니다.

```bash 
git clone https://github.com/openclaw/openclaw.git
cd openclaw
```

## 도커 셋업 스크립트 실행

`openclaw` 폴더 안으로 이동한 상태에서, 아래 스크립트를 실행하여 도커(Docker) 환경 구성을 시작합니다. 이 명령어는 필요한 도커 이미지를 다운로드하고 기본 설정 파일을 생성해 줍니다.

```bash
./docker-setup.sh
```

![[Pasted image 20260228142002.png]]

### Caustion

Windows 환경의 Git Bash에서는 간혹 방향키(↑, ↓)로 옵션을 선택하는 인터랙티브(Interactive) UI가 정상적으로 작동하지 않을 수 있습니다. 화살표를 눌러도 선택지가 움직이지 않는다면, 당황하지 말고 아래의 두 가지 방법 중 하나로 해결하세요.

**해결 방법 1: Git 버전 최신으로 업그레이드하기**
현재 열려있는 Git Bash 창에 아래 명령어를 입력해 Git 자체를 업데이트합니다.

```bash
git update-git-for-windows
```
 **Note:** 업데이트가 완료되면 Git Bash를 껐다가 다시 켠 후, `./docker-setup.sh` 명령어를 다시 실행해 봅니다.

**해결 방법 2: CMD(명령 프롬프트)에서 수동으로 실행하기**
Git 업데이트로도 해결되지 않는다면, Windows 기본 터미널인 cmd를 사용해야 합니다.

1. 현재 먹통이 된 Git Bash 창은 종료합니다.
    
2. 윈도우 **파일 탐색기**를 열고, 아까 클론 받은 `openclaw` 폴더 안으로 들어갑니다.
    
3. 탐색기 상단의 경로 표시줄(주소창) 을 클릭하고, 기존 주소를 지운 뒤 `cmd`라고 타이핑하고 `Enter`를 누릅니다. (해당 폴더 경로가 잡힌 상태로 cmd 창이 곧바로 열립니다.)
    
4. 새로 열린 cmd 창에 아래 명령어를 복사해서 붙여넣고 실행하여 온보딩 위자드를 강제로 시작합니다.

```bash
docker compose run --rm openclaw-cli onboard
```

## 온보드 위자드 (Onboard Wizard) 설치 진행

터미널 화면에 나타나는 질문들에 맞춰 초기 환경을 세팅합니다. (방향키로 이동하고 `Enter`로 선택합니다.)

```bash
◆  I understand this is personal-by-default and shared/multi-user use requires lock-down. Continue?
│  ○ Yes / ● No
└
```

- **선택:** `Yes`
- **설명:** OpenClaw는 기본적으로 개인용으로 세팅되어 있습니다. 외부 서버 등 여러 명이 공유하는 환경이라면 추가 보안 잠금이 필요하다는 안내입니다. 개인 로컬 환경에서 도커로 구동하는 것이므로 동의(Yes)하고 넘어갑니다.

```bash
◆  Onboarding mode
│  ● QuickStart (Configure details later via openclaw configure.)
│  ○ Manual
└
```

- **선택:** `QuickStart`
- **설명:** 필수적인 설정만 빠르게 진행하는 모드입니다. 복잡한 워크스페이스(Workspace) 구성이나 세부 설정은 나중에 설정 파일이나 대시보드에서 수정할 수 있으니 퀵스타트를 선택합니다.

```bash
◆  Model/auth provider
│  ● OpenAI (Codex OAuth + API key)
│  ○ Anthropic
│  ○ Chutes
│  ○ vLLM
│  ○ MiniMax
│  ○ Moonshot AI (Kimi K2.5)
│  ○ Google
│  ○ xAI (Grok)
│  ○ Mistral AI
│  ○ Volcano Engine
│  ○ BytePlus
│  ○ OpenRouter
│  ○ Kilo Gateway
│  ○ Qwen
│  ○ Z.AI
│  ○ Qianfan
│  ○ Copilot
│  ○ Vercel AI Gateway
│  ○ OpenCode Zen
│  ○ Xiaomi
│  ○ Synthetic
│  ○ Together AI
│  ○ Hugging Face
│  ...
└
```

- **선택:** `Google`
- **설명:** OpenClaw의 두뇌 역할을 할 AI 제공자를 선택합니다. 우리는 Gemini를 사용할 것이므로 Google을 선택합니다.

```bash
◆  Google auth method
│  ● Google Gemini API key
│  ○ Google Gemini CLI OAuth
│  ○ Back
└
```

- **선택:** `Google Gemini API Key`
- **설명:** 발급받은 API 키를 직접 입력하는 방식이 도커 환경에서 관리하고 연동하기 가장 직관적입니다.

```
◆  How do you want to provide this API key?
│  ● Paste API key now (Stores the key directly in OpenClaw config)
│  ○ Use secret reference
└
```

- **선택:** `Paste API key now`
- **설명:** 터미널에 바로 키를 붙여넣어 OpenClaw 설정 파일에 저장합니다.
    

> **Google AI Studio API 발급 및 무료 티어 팁**
> 
> - **발급처:** API 키가 없다면 [Google AI Studio API Key 페이지](https://aistudio.google.com/app/apikey)에 접속하여 Google 계정 로그인 후 'Create API Key'를 눌러 무료로 발급받으세요.

![[스크린샷 2026-02-28 142413.png]]

> - **무료 티어 주의사항:** 신용카드를 등록하지 않은 무료 환경(Free Tier)에서는 모델별로 분당 요청 횟수(RPM)나 하루 할당량 제한이 있습니다. OpenClaw 사용 중 갑자기 응답을 안 하거나 에러를 뱉는다면 이 무료 한도에 도달했을 확률이 높으니, AI Studio 대시보드에서 잔여량을 체크해 보는 것이 좋습니다.

- **입력 방법:** 복사한 API 키를 터미널 창에 붙여넣습니다. (터미널에서는 `Ctrl+V`가 안 먹힐 수 있으니, 마우스 우클릭 후 'Paste'를 누르거나 `Shift + Insert` 키를 사용하세요.) 붙여넣은 후 `Enter`를 누릅니다.


```bash
◆  Default model
│  ● Keep current (google/gemini-3-pro-preview)
│  ○ Enter model manually
│  ○ google/gemini-1.5-flash
│  ○ google/gemini-1.5-flash-8b
│  ○ google/gemini-1.5-pro
│  ○ google/gemini-2.0-flash
│  ○ google/gemini-2.0-flash-lite
│  ○ google/gemini-2.5-flash
│  ○ google/gemini-2.5-flash-lite
│  ○ google/gemini-2.5-flash-lite-preview-06-17
│  ○ google/gemini-2.5-flash-lite-preview-09-2025
│  ○ google/gemini-2.5-flash-preview-04-17
│  ○ google/gemini-2.5-flash-preview-05-20
│  ○ google/gemini-2.5-flash-preview-09-2025
│  ○ google/gemini-2.5-pro
│  ○ google/gemini-2.5-pro-preview-05-06
│  ○ google/gemini-2.5-pro-preview-06-05
│  ○ google/gemini-3-flash-preview
│  ○ google/gemini-3-pro-preview
│  ○ google/gemini-3.1-pro-preview
│  ○ google/gemini-3.1-pro-preview-customtools
│  ○ google/gemini-flash-latest
│  ○ google/gemini-flash-lite-latest
│  ...
└
```

- **선택:** 원하는 모델 (예: `google/gemini-3.1-pro-preview` 등)을 선택.
- **설명:** 기본으로 사용할 모델을 지정합니다. 복잡한 코딩이나 에이전트 작업에는 **Pro** 모델이 좋고, 단순하고 빠른 응답이 필요할 때는 **Flash** 모델을 선택하는 것이 유리합니다. (목록에 최신 모델이 있다면 선택해 주세요.)

```
◆  Select channel (QuickStart)
│  ● Telegram (Bot API) (recommended · newcomer-friendly)
│  ○ WhatsApp (QR link)
│  ○ Discord (Bot API)
│  ○ IRC (Server + Nick)
│  ○ Google Chat (Chat API)
│  ○ Slack (Socket Mode)
│  ○ Signal (signal-cli)
│  ○ iMessage (imsg)
│  ○ Feishu/Lark (飞书)
│  ○ Nostr (NIP-04 DMs)
│  ○ Microsoft Teams (Bot Framework)
│  ○ Mattermost (plugin)
│  ○ Nextcloud Talk (self-hosted)
│  ○ Matrix (plugin)
│  ○ BlueBubbles (macOS app)
│  ○ LINE (Messaging API)
│  ○ Zalo (Bot API)
│  ○ Zalo (Personal Account)
│  ○ Synology Chat (Webhook)
│  ○ Tlon (Urbit)
│  ○ Skip for now
└
```

- **선택:** `Skip for now`
- **설명:** 지금 당장 메신저와 연결하지 않고 초기 설치를 먼저 마무리합니다. 텔레그램(Telegram) 연동 등은 설치 완료 후 워크스페이스 설정 단계에서 봇 토큰(Bot Token)을 발급받아 따로 진행하는 것이 훨씬 깔끔합니다.

```
◇  Select channel (QuickStart)
│  Skip for now
Updated ~/.openclaw/openclaw.json
Workspace OK: ~/.openclaw/workspace
Sessions OK: ~/.openclaw/agents/main/sessions
│
◇  Skills status ─────────────╮
│                             │
│  Eligible: 3                │
│  Missing requirements: 41   │
│  Unsupported on this OS: 7  │
│  Blocked by allowlist: 0    │
│                             │
├─────────────────────────────╯
│
◆  Configure skills now? (recommended)
│  ● Yes / ○ No
└
```

- **선택:** `No`
- **설명:** 스킬은 OpenClaw가 웹 검색을 하거나 터미널 명령을 내리는 등의 추가 능력을 부여하는 플러그인입니다. 초기 설치 단계에서 이것저것 켜두면 충돌이 날 수 있으니 일단 건너뛰고, 나중에 대시보드에서 필요한 것만 활성화합니다.

```
◇  Hooks ──────────────────────────────────────────────────────────────────╮
│                                                                          │
│  Hooks let you automate actions when agent commands are issued.          │
│  Example: Save session context to memory when you issue /new or /reset.  │
│                                                                          │
│  Learn more: https://docs.openclaw.ai/automation/hooks                   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────╯
│
◆  Enable hooks?
│  ◻ Skip for now
│  ◻ 🚀 boot-md
│  ◻ 📎 bootstrap-extra-files
│  ◻ 📝 command-logger
│  ◻ 💾 session-memory
└
```

- **선택:** `Skip for now`를 제외한 나머지 4개 항목을 **모두 체크**한 뒤 `Enter`.
    
- **설명:** 특정 상황에서 자동으로 작동하는 편의 기능들입니다. 특히 `session-memory`는 AI가 이전 대화의 맥락을 기억하게 해주는 핵심 기능이므로 꼭 켜두는 것이 좋습니다.
    
- **조작법:** 방향키로 위아래로 이동하며 스페이스바(Spacebar)를 누르면 체크가 됩니다. 4개를 모두 체크했다면 `Enter`를 누릅니다.

```
◇  What now ─────────────────────────────────────────────────────────────╮
│                                                                        │
│  What now: https://openclaw.ai/showcase ("What People Are Building").  │
│                                                                        │
├────────────────────────────────────────────────────────────────────────╯
│
└  Onboarding complete. Use the dashboard link above to control OpenClaw.
```

## Gateway 토큰 확인 및 .env 업데이트

OpenClaw 대시보드(웹 UI)에 안전하게 접속하기 위해서는 보안 토큰(Token)을 환경 변수 파일(`.env`)에 등록해 주어야 합니다.

<font color="#ffff00">1) 토큰 발급 및 확인 명령어 실행</font>

Git Bash(또는 cmd) 터미널에서 아래 명령어를 입력하여 내 OpenClaw 전용 대시보드 URL과 토큰을 생성합니다.

```bash
docker compose run --rm openclaw-cli dashboard --no-open
```

명령어를 실행하면 터미널에 아래와 비슷한 결과가 출력됩니다.

<font color="#ffff00">2) 터미널 출력 결과 확인</font>

```
Dashboard URL: http://127.0.0.1:18789/#token=MYTOKEN12345abcdef
Copy to clipboard unavailable.
Browser launch disabled (--no-open). Use the URL above.
```

여기서 `#token=` 뒤에 있는 값(**예시에서는 `MYTOKEN12345abcdef`**)이 바로 복사해야 할 토큰 값입니다. 이 값을 드래그해서 복사해 둡니다.

`openclaw` 폴더 안에 있는 `.env` 파일을 메모장이나 코드 에디터(VS Code 등)로 엽니다.

- _Tip: 터미널에서 `notepad .env`를 입력하면 윈도우 메모장으로 바로 열 수 있습니다._

<font color="#ffff00">3) .env 파일 수정 및 경로 이해하기</font>

`openclaw` 폴더 안에 있는 `.env` 파일을 메모장이나 코드 에디터(VS Code 등)로 엽니다.

파일을 열면 설치 시 자동으로 채워진 값들이 있습니다. 여기서 주목해야 할 부분은 `OPENCLAW_CONFIG_DIR`과 `OPENCLAW_WORKSPACE_DIR`입니다. 이 두 경로는 사용 중인 **운영체제(OS)에 따라 형태가 다릅니다.**

- **Windows (Git Bash 환경 기본값):** `C:\` 드라이브가 `/c/` 형태로 표기됩니다.
    - `OPENCLAW_CONFIG_DIR=/c/Users/사용자명/.openclaw`
    - `OPENCLAW_WORKSPACE_DIR=/c/Users/사용자명/.openclaw/workspace`
        
- **Linux 기본값:** 리눅스는 드라이브 개념 대신 `/home` 폴더를 사용합니다.
    - `OPENCLAW_CONFIG_DIR=/home/사용자명/.openclaw`
    - `OPENCLAW_WORKSPACE_DIR=/home/사용자명/.openclaw/workspace`
        
    - _(참고: macOS의 경우 `/Users/사용자명/.openclaw` 형태가 됩니다.)_

방금 복사한 토큰 값을 `OPENCLAW_GATEWAY_TOKEN=` 뒤에 붙여넣고 **저장**합니다.

```
# 수정 후 예시 (.env 파일 내부)
OPENCLAW_CONFIG_DIR=/c/Users/사용자명/.openclaw
OPENCLAW_WORKSPACE_DIR=/c/Users/사용자명/.openclaw/workspace
OPENCLAW_GATEWAY_PORT=18789
OPENCLAW_BRIDGE_PORT=18790
OPENCLAW_GATEWAY_BIND=lan
OPENCLAW_GATEWAY_TOKEN=MYTOKEN12345abcdef
# ... (이하 생략)
```

<font color="#00b050">Config 및 Workspace 폴더 위치를 다른 곳으로 옮기고 싶다면?</font>

기본적으로 `C 드라이브` (또는 홈 디렉터리)에 저장되는 설정 파일과 워크스페이스(작업 공간) 용량이 커질 것을 대비해, `D 드라이브`나 다른 외장 폴더로 위치를 옮기고 싶을 수 있습니다. 이럴 때는 **실제 폴더를 복사/이동한 뒤, `.env` 파일의 경로만 수정해 주면 됩니다.**

## Gateway 시작 및 로그 확인

토큰과 경로 설정이 모두 끝났으니, 이제 게이트웨이 컨테이너를 실행할 차례입니다

<font color="#ffff00">1) Gateway 컨테이너 백그라운드 실행</font>

```bash
docker compose up -d openclaw-gateway
```

`-d` (detached) 옵션은 컨테이너를 백그라운드에서 실행하라는 뜻입니다. 이 옵션을 빼면 터미널 창을 끄는 순간 OpenClaw도 같이 꺼집니다.

<font color="#ffff00">2) 도커 로그 확인 (선택사항)</font>

컨테이너가 정상적으로 켜졌는지, 에러가 발생하지 않았는지 확인하려면 아래 명령어로 로그를 봅니다.

```bash
docker compose logs -f openclaw-gateway
```

`-f` (follow) 옵션을 넣으면 로그가 실시간으로 출력됩니다. 로그 보기를 종료하고 싶을 때는 **`Ctrl + C`** 를 누르면 다시 명령어 입력창으로 빠져나옵니다.

### 트러블슈팅: Control UI 권한 에러 해결법

로그를 확인했을 때, 컨테이너가 정상적으로 실행되지 않고 아래와 같은 에러 메시지가 출력될 수 있습니다.

```bash
gateway failed to start: Error: non-loopback Control UI requires gateway.controlUi.allowedOrigins (set explicit origins), or set gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback=true to use Host-header origin fallback mode
```

OpenClaw의 보안 정책상, 명시적으로 허용된 주소(Origin)가 아니면 웹 대시보드 접근을 차단하기 때문에 발생하는 에러입니다.

**해결 방법:**

1. 이전에 설정한 `OPENCLAW_CONFIG_DIR` 경로 안에 있는 `.openclaw/openclaw.json` 설정 파일을 메모장이나 코드 에디터로 엽니다.
    
2. 파일 내용 중 `"gateway": { ... }` 부분을 찾아서, 아래와 같이 `"controlUi"` 항목과 허용할 주소(`allowedOrigins`)를 추가해 줍니다.

```json
{
    // ... (다른 설정들) ...
    "gateway": {
        "controlUi": {
            "allowedOrigins": ["http://127.0.0.1:18789", "http://localhost:18789"]
        },
        // ... (기존 gateway 내부 설정들) ...
    }
}
```

3. 파일을 저장한 후, 터미널로 돌아가 **게이트웨이를 재시작**하여 변경된 설정을 적용합니다.

```bash
docker compose restart openclaw-gateway
```


## 웹 대시보드 접속 및 디바이스 페어링

게이트웨이 컨테이너가 정상적으로 실행되었다면, 브라우저를 열고 설정한 포트로 대시보드에 접속합니다.

<font color="#ffff00">1) 웹 대시보드 접속</font>
브라우저 주소창에 아래 URL을 입력하여 OpenClaw 대시보드로 이동합니다.

```
http://localhost:18789
```

![[Pasted image 20260228150217.png|697]]

<font color="#ffff00">2) Gateway Token 일치 여부 확인</font>

대시보드 화면의 **Overview** 탭에서 현재 웹 페이지가 인식하고 있는 **Gateway Token** 값이, 우리가 앞서 `.env` 파일의 `OPENCLAW_GATEWAY_TOKEN`에 입력했던 값과 일치하는지 확인합니다.

![[스크린샷 2026-02-28 150511.png]]

<font color="#00b050">토큰이 일치하지 않거나 연결이 안 될 때 해결법</font>

- **방법 A (웹에서 새로고침):** 우리가 `.env`에 적었던 토큰 값을 복사한 뒤, 웹 브라우저의 Token 입력 칸에 직접 붙여넣고 `Refresh` 또는 `Save` 버튼을 클릭합니다.

- **방법 B (도커 컨테이너 재생성):** `.env` 파일의 내용을 수정했다면 단순한 `restart` 명령어로는 변경된 값이 적용되지 않습니다. 도커가 수정된 환경 변수를 새로 읽어들일 수 있도록 컨테이너를 완전히 내렸다가 다시 올려야 합니다.

```
docker compose down openclaw-gateway
```

```
docker compose up -d openclaw-gateway
```

### 디바이스 페어링 승인

웹 대시보드에서 게이트웨이를 제어하려면, 현재 접속한 브라우저(디바이스)를 신뢰할 수 있도록 터미널에서 직접 **승인(Approve)** 을 해줘야 합니다.

도커 환경에서는 이 작업을 할 때 `docker compose run`이 아닌 **`docker compose exec`** 를 사용해야 합니다.

- `run`: 완전히 새로운 컨테이너를 임시로 하나 더 띄워서 명령어를 실행합니다. (여기서는 페어링 정보가 엇갈릴 수 있음)
    
- `exec`: **현재 실행 중인** `openclaw-gateway` 컨테이너 내부로 들어가서 명령어를 실행합니다. (이 방식을 써야 정확히 페어링됩니다.)

웹 대시보드에서 페어링 요청을 보낸 상태로 두고, 터미널(Git Bash 또는 PowerShell)에 아래 명령어를 입력하여 대기 중인 요청 목록을 확인합니다.

```
docker compose exec openclaw-gateway node dist/index.js devices list	
```

![[스크린샷 2026-02-28 150939.png]]

명령어를 치면 터미널에 표 형태로 기기 목록이 뜹니다. 그중 방금 웹에서 요청한 항목을 찾아, **`Request` 열(Column) 아래에 있는 고유 ID 값 (예: `fb9d...` 로 시작하는 문자열)** 을 드래그해서 복사합니다.

```
docker compose exec openclaw-gateway node dist/index.js devices approve fb9dxxx-xxxx-xx
```

명령어가 성공적으로 실행되면 터미널에 아래와 같은 승인 완료 메시지가 뜹니다.

```
│
◇
Approved a4379xxxxxxxxx (fb9dxxx-xxxx-xxxx)
```


### 페어링 목록이 뜨지 않을 경우 A (클라우드 환경)

1. 클라우드 환경에서 `devices list` 에 항목이 뜨지 않을 경우 다음과 같은 원인일 가능성이 큽니다.
	- **외부 IP 접속 문제** : 브라우저에서 `http:<클라우드ip>:<포트(기본 18789)>` 로 직접 접속하면 게트웨이는 내부 localhost 기반 요청으로 인식하지 않습니다.
	- Gateway는 기본적으로 `localhost` 기준 컨텍스트에서 페어링을 처리하도록 설계되어 있어, 외부 IP 기반 접근 시 정상 동작하지 않을 수 있습니다.

![[Pasted image 20260304165259.png]]

이 경우, **SSH 포트 포워딩을 통해 로컬 localhost로 접속하도록 우회하면 해결**됩니다.

로컬 PC에서 다음 명령어를 실행합니다 (VS Code로 SSH 를 접속하면 접속된 환경에서도 가능합니다.)

```bash
ssh -L 18789:localhost:18789 user@remote_server_ip
```

이후 브라우저로 접속합니다.

```
http://localhost:18789
```

그 다음 페어링 목록을 확인하고 승인합니다.

```docker
docker compose exec openclaw-gateway node dist/index.js devices list
```

```
docker compose exec openclaw-gateway node dist/index.js devices approve <requestID>
```
### 페어링 목록이 뜨지 않을 경우 B

1. `devices list` 에 항목이 뜨지 않을 경우 다음과 같은 원인일 가능성이 큽니다.
	- **비보안 컨텍스트(Insecure Context):** 브라우저가 `https`가 아닌 `http`로 접속된 외부 IP를 '위험'으로 간주하여, 기기 식별값 자체를 게이트웨이에 전송하지 않는 경우입니다. 이 경우 서버는 요청을 받은 적이 없으므로 목록에 나타나지 않습니다.
	- **토큰/포트 불일치:** CLI 명령어가 사용하는 기본 포트(18789)와 실제 설정한 포트가 다를 때 게이트웨이와 통신이 되지 않아 아무 결과도 출력되지 않습니다.
2. 위의 페어링 과정이 정상적으로 진행되지 않을 경우, 보안 설정을 한 단계 낮추어 즉시 접속할 수 있습니다.

설정 파일 수정 (`.openclaw/openclaw.json`):

```json
"gateway": {
  ...
  "controlUi": {
    "enabled": true,
    "allowInsecureAuth": true,
    "dangerouslyDisableDeviceAuth": true,  // 이 옵션을 true로 추가
    "allowedOrigins": ["*"]
  }
}
```

3. 이 방식은 편리하지만 다음과 같은 보안 취약점이 발생하므로 주의해야 합니다.
	- **기기 식별 불가:** 어떤 기기가 접속했는지 서버가 검증하지 않습니다. 즉, **Gateway Token만 유출되면** 전 세계 누구라도 사용자님의 서버에 접속해 API를 남용하거나 대화 내역을 볼 수 있습니다.
	- **무단 접속 위험:** 브루트포스(무차별 대입) 공격으로 토큰이 뚫릴 경우, 2차 방어선(기기 승인)이 없기 때문에 서버가 완전히 노출됩니다.



이제 다시 웹 브라우저 대시보드로 돌아가 보면, 화면이 새로고침 되면서 게이트웨이와 정상적으로 **페어링(Connected)** 이 완료된 화면을 보실 수 있습니다

![[Pasted image 20260228151329.png]]

## 설정 변경 (lan 모드로 전환)

페어링이 성공적으로 완료되었다면, 이제 CLI 명령어를 편하게 사용하고 외부 봇(카카오톡 등)과 원활하게 통신하기 위해 설정을 변경합니다.

1. `.openclaw/openclaw.json` 원본 파일을 메모장으로 엽니다.
2. `gateway` 섹션의 `"bind": "loopback"` 부분을 찾아서 **`"bind": "lan"`** 으로 변경하고 저장합니다. (약 10~20번째 줄)

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

> **왜 바꾸나요?**  
> 페어링은 보안 문제로 `loopback` 상태에서 해야 잘 되지만, 이후 다른 명령어(`openclaw-cli`)나 카카오톡 연동은 `lan` 모드에서 훨씬 간편하게 작동하기 때문입니다. 이미 페어링된 정보는 사라지지 않으니 안심하세요



