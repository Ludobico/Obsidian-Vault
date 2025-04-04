![[Pasted image 20250404102922.png|256]]

[original source](https://tiaz.dev/ai/2)

Cursor는 AI를 활용한 코드 편집기(에디터)로, 개발자의 생산성 향상에 초점을 맞춘 도구입니다. VSCode 기반으로 만들어져있기 때문에 VScode 설정을 그대로 마이그레이션해서 사용 할 수 있습니다.

### Cursor 설치

[Download Cursor](https://www.cursor.com/downloads) 페이지에서 Cursor를 다운받아 설치해 줍니다.

## Cursor MCP

Cursor는 [[MCP(Model Context Protocol)]]를 기본적으로 지원합니다. 이를 통해 AI와 다양한 서비스를 신속하고 간편하게 연동할 수 있습니다.

1. `Open Cursor Settings` 버튼 클릭
2. `MCP` 탭 선택
3. 원하는 MCP 설정

![[Pasted image 20250404102955.png]]

### MCP servers

여기에서는 두 가지 MCP 서버를 연동해보겠습니다.

- [GitHub MCP Server](https://smithery.ai/server/@smithery-ai/github)
- [BrowserTools MCP](https://github.com/AgentDeskAI/browser-tools-mcp)

## GitHub 연동

### Smithery

[Smithery](https://smithery.ai/)는 MCP 서버를 모아둔 사이트 입니다. 다양한 MCP 서버가 있으니 필요한 MCP 서버를 찾아서 사용해 보세요!

![[Pasted image 20250404103011.png]]

### GitHub MCP Server

Smithery에서 GitHub로 검색하면 `@smithery-ai/github`의 GitHub를 클릭합니다.

![[Pasted image 20250404103046.png]]

### GitHub 토큰

MCP 서버가 GitHub에 접근하려면 GitHub 토큰이 필요합니다.

1. GitHub 프로필에서 `Settings` 클릭
2. `Developer Settings` 탭 클릭 (가장 하단에 있음)
3. `Personal access tokens` - `Fine-grained tokens` - `Generate new token` 버튼 클릭

![[Pasted image 20250404103059.png]]

### 토큰 권한 설정 및 생성

> 토큰은 유출되지 않게 복사해두고 잘 간직합니다!

아래의 예시와 같이 필요한 권한을 설정해서 토큰을 생성합니다.

- Expiration : 토큰 만료기간 설정
- Repository access : All repositories
- Repository permissions : 필요한 기능을 사용할 수 있게 `Read and write` 권한을 설정
    - Actions
    - Administration
    - Commit statuses
    - Contents
    - Issues
    - Commit statuses

### Smithery Installation

> Cursor 버전을 꼭 확인 하세요! Cursor ≥0.47.x 버전에 따라 설정 방법에 차이가 있습니다!


1. Smithery 우측에 Installation `Cursor` 탭 클릭
2. GitHub 토큰을 입력하고 `Connect` 클릭
3. Cursor 버전과 운영체제에 알맞은 커맨드를 복사 또는 JSON으로 복사

![[Pasted image 20250404103147.png]]

### Cursor에 적용 (Cursor ≤0.46.x)

> 필요에 따라서 Docker나 node, python등 추가 설치가 필요합니다. 꼭 커맨드를 확인하세요!

이제 복사한 커맨드를 설정만하면 Cursor에 MCP 서버 설정이 끝납니다!

1. Cursor에서 MCP 탭 `Add new MCP server` 클릭
2. `Name` : 아무 이름이나 상관없음
3. `Type` : command 선택
4. `Command` : 복사한 커맨드 붙여넣기

![[Pasted image 20250404103210.png]]

### mcp.json (Cursor ≥0.47.x)

> mcp 서버를 추가하는 UI 팝업이 안나오는 경우, 직접 JSON으로 설정해보세요!

`mcp.json` 파일을 통해 각 프로젝트별, 또는 전체 프로젝트에서 실행 할 MCP 서버를 설정 할 수 있습니다. [Configuration Locations](https://docs.cursor.com/context/model-context-protocol#configuration-locations) 페이지를 참고하시길 바랍니다!

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@smithery-ai/github",
        "--config",
        "{\"githubPersonalAccessToken\":\".............\"}"
      ]
    }
  }
}
```

### Cursor! GitHub 레포 만들어줘

> MCP는 현재(2025-03-27) Anthropic 모델에만 지원합니다. 따라서 Cursor에서 Claude 모델을 선택해야 합니다. `Big News` OpenAI에서도 MCP 지원을 공식 발표!

이제 Cursor에게 GitHub 관련된 작업을 시킬 수 있습니다. Cursor가 필요한 동작을 수행 할 수 있게 `Run tool`, `Accept`만 해주면 됩니다!

![[Pasted image 20250404103258.png]]

