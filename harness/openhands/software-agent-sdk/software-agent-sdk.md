[[openhands]] SDK는 현재 Python SDK와 Agent Server뿐 아니라 agents, tools, conversations, workspaces, events 및 browser-compatible TypeScript client까지 포함합니다.

## Architecture

### 1. Agent (에이전트)
에이전트는 실행 루프, 도구(Tool) 호출 및 상태 전환을 관리하는 핵심 오케스트레이터입니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/agent/`
- **주요 파일:**
    - `agent.py`: 핵심 실행 로직(`step`, `astep`)을 구현한 `Agent` 클래스 포함.
    - `base.py`: `AgentBase` 추상 클래스 정의.
    - `parallel_executor.py`: 병렬 도구 실행 관리.
- **주요 클래스:** `Agent`, `AgentBase`, `ParallelToolExecutor`

### 2. Conversation (대화)

대화 영역은 에이전트 상호작용의 상태, 이력, 생명주기를 관리합니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/conversation/`
- **주요 파일:**
    - `conversation.py`: `Conversation` 클래스 및 팩토리 정의.
    - `state.py`: `ConversationState`(실행 상태, 브랜치, 이벤트 등) 관리.
    - `event_store.py`: 대화 이벤트의 영속성(persistence) 처리.
- **주요 클래스:** `Conversation`, `ConversationState`, `LocalConversation`, `RemoteConversation`

### 3. LLM (언어 모델)

언어 모델과의 통신을 담당하며, 라우팅, 메시지 포맷팅, 응답 파싱을 처리합니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/llm/`
- **주요 파일:**
    - `llm.py`: 응답 생성을 위한 기본 `LLM` 클래스.
    - `message.py`: `Message` 구조 정의.
    - `router/`: 모델 라우팅 로직 (`base.py`, `impl/`).
- **주요 클래스:** `LLM`, `Message`, `RouterLLM`

### 4. Tools (도구)

에이전트가 사용할 수 있는 기능(Capability)들입니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/tool/` 및 `openhands-tools/openhands/tools/`
- **주요 파일:**
    - `tool.py`: 기본 `Tool` 클래스 및 정의.
    - `registry.py`: 도구 발견 및 가용성 관리.
    - `builtins/`: `FinishTool`, `ThinkAction`, `TerminalTool` 등 핵심 도구 포함.
- **주요 클래스:** `Tool`, `ToolDefinition`, `MCPToolDefinition`

### 5. Context (컨텍스트)

시스템 프롬프트와 메모리를 포함하여 에이전트에게 제공되는 정보를 관리합니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/context/`
- **주요 파일:**
    - `agent_context.py`: `AgentContext` 모델 정의.
    - `prompts/`: 시스템 프롬프트 템플릿 및 레지스트리 관리.
    - `condenser/`: 컨텍스트 윈도우에 맞게 대화 이력을 압축하는 로직.
- **주요 클래스:** `AgentContext`, `Condenser`

### 6. Events (이벤트)

대화 내에서 발생하는 액션, 관찰(Observation), 메시지의 흐름을 나타냅니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/event/`
- **주요 파일:**
    - `base.py`: 기본 이벤트 클래스.
    - `llm_convertible/`: LLM 메시지로 변환 가능한 이벤트 정의 (예: `ActionEvent`, `ObservationEvent`).
- **주요 클래스:** `Event`, `ActionEvent`, `ObservationEvent`, `MessageEvent`, `SystemPromptEvent`

### 7. Workspace (워크스페이스)

에이전트가 액션을 수행하는 환경(파일 시스템, 터미널 등)입니다.

- **디렉토리:** `openhands-sdk/openhands/sdk/workspace/` 및 `openhands-workspace/openhands/workspace/`
- **주요 파일:**
    - `workspace.py`: 기본 `Workspace` 클래스.
    - `local.py` / `remote.py`: 다양한 런타임 환경을 위한 구현체.
- **주요 클래스:** `Workspace`, `LocalWorkspace`, `RemoteWorkspace`, `AgentSandboxWorkspace`


