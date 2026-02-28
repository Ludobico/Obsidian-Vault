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

Git Bash(또는 cmd) 터미널에서 아래 명령어를 입력하여 내 OpenClaw 전용 대시보드 URL과 토큰을 생성합니다.

```bash
docker compose run --rm openclaw-cli dashboard --no-open
```

명령어를 실행하면 터미널에 아래와 비슷한 결과가 출력됩니다.

```
Dashboard URL: http://127.0.0.1:18789/#token=MYTOKEN12345abcdef
Copy to clipboard unavailable.
Browser launch disabled (--no-open). Use the URL above.
```

여기서 `#token=` 뒤에 있는 값(**예시에서는 `MYTOKEN12345abcdef`**)이 바로 복사해야 할 토큰 값입니다. 이 값을 드래그해서 복사해 둡니다.

`openclaw` 폴더 안에 있는 `.env` 파일을 메모장이나 코드 에디터(VS Code 등)로 엽니다.

- _Tip: 터미널에서 `notepad .env`를 입력하면 윈도우 메모장으로 바로 열 수 있습니다._

파일을 열면 아래와 같은 항목들이 있습니다.

```
OPENCLAW_CONFIG_DIR=
OPENCLAW_WORKSPACE_DIR=
OPENCLAW_GATEWAY_PORT=18789
OPENCLAW_BRIDGE_PORT=18790
OPENCLAW_GATEWAY_BIND=lan
OPENCLAW_GATEWAY_TOKEN=
OPENCLAW_IMAGE=openclaw:local
OPENCLAW_EXTRA_MOUNTS=
OPENCLAW_HOME_VOLUME=
OPENCLAW_DOCKER_APT_PACKAGES=
```

방금 복사한 토큰 값을 `OPENCLAW_GATEWAY_TOKEN=` 뒤에 붙여넣고 **저장**합니다.

```
OPENCLAW_GATEWAY_TOKEN=MYTOKEN12345abcdef
```

