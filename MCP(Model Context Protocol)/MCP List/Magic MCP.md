
[smithery](https://smithery.ai/server/@21st-dev/magic-mcp?code=3bb45f4d-d3cd-4695-8eb4-3f27214caefb)

[github](https://github.com/21st-dev/magic-mcp?tab=readme-ov-file)

# 21st.dev Magic AI Agent

<font color="#ffc000">Magic Component Platform (MCP)</font>는 개발자가 자연올 UI 컴포넌트를 요청하면, 즉시 **UI 컴포넌트를 생성해주는 AI 기반 도구**입니다. UI 디자인과 개발을 간소화하고, 빠르게 원하는 컴포넌트를 얻을 수 있도록 설계되었습니다.

## How it works

### 1. 자연어 기반 컴포넌트 생성
	- 채팅창에서 `/ui` 명령어를 사용해 원하는 UI를 설명
	- Example : `/ui create a modern navigation bar with responsive design`
	- 설명을 기반으로 AI가 자동으로 UI 컴포넌트를 생성

### 2. IDE와의 연동
	- 다음과 같은 IDE에서 작동
		- Cursor ([[Apply MCP in cursorAI]]) 참조
		- Windsurf
		- VSCode (Cline extension 설치 필요)

### 3. 21st.dev UI 라이브러리를 기반으로 디자인
	- AI가 생성해내는 컴포넌트는 21st.dev의 디자인 시스템을 참고해 일관된 스타일을 유지


## Getting Started

### 1. 사전 준비

- **Node.js (최신 LTS 버전)** 필요
    
- 지원 IDE 중 하나 설치:
    
    - Cursor
        
    - Windsurf
        
    - VSCode (Cline extension 필수)
        

### 2. 설치 및 API Key 발급

- [21st.dev Magic Console](https://21st.dev) 방문
    
- API Key 생성

## FAQ

### How does Magic AI Agent handle my codebase?

[[MCP(Model Context Protocol)]] 의 Magic AI Agent 는 프로젝트 내에서 컴포넌트와 관련된 파일만 생성하거나 수정합니다. 기존 코드 스타일을 유지하며, 프로젝트의 구조를 따릅니다.

### Can i Customize the generated components?

모든 컴포넌트를 편집 가능하며, 코드 구조도 잘 정리되어 있습니다.

### What happens if  i run out of generations?

월별 생성 제한 초과 시, 요금제 업그레이드 요청이 표시됩니다.
기존 생성된 컴포넌트는 그대로 유지되며, 추가 생성만 제한됩니다.


## Development Structure

MCP의 프로젝트 구조는 다음과 같이 구성됩니다.

```
mcp/
├── app/
│   └── components/     # 핵심 UI 컴포넌트 폴더
├── types/             # TypeScript 타입 정의
├── lib/              # 유틸리티 함수 모음
└── public/           # 정적(Static) 파일 및 자산
```

