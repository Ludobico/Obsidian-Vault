[[Antigravity CLI]] 는 커스텀 단축키를 지원합니다.

- 단축키 변경
	- CLI 내에서 `/keybindings` 명령어를 입력하거나 JSON 파일을 직접 수정
- 설정 파일 위치
	- `~/.gemini/antigravity-cli/keybindings.json`
- 초기화
	- 기본값으로 되돌리려면 `keybindings.json` 파일을 삭제하면 됩니다.

## Default Keybindings

| Action           | Keys                                     | Description                 |
| ---------------- | ---------------------------------------- | --------------------------- |
| Clear TUI Screen | Ctrl + L                                 | 터미널 출력 화면 지우기               |
| Enter            | Enter                                    | 프롬프트 전송                     |
| Escape           | Ctrl + C, Esc                            | 응답 생성 중지, 입력창 초기화           |
| Exit CLI         | Ctrl + D                                 | CLI 종료                      |
| Suspend CLI      | Ctrl + Z                                 | CLI 작업 일시 정지(백그라운드로 전환)     |
| Edit Command     | E                                        | 에디터 열기                      |
| Confirm No       | N                                        | 명령어 실행 취소                   |
| Confirm Yes      | Y                                        | 명령어 실행 승인                   |
| Open Editor      | Ctrl + G                                 | 외부 에디터(vi, nano 등)로 프롬프트 작성 |
| Paste Text       | Ctrl + V                                 | 텍스트 붙여넣기                    |
| Redo Text Edit   | Ctrl + Shift + Z                         | 실행 취소 되돌리기 (redo)           |
| Undo Text Edit   | Ctrl + _                                 | 텍스트 입력 이전 상태로 되돌리기 (undo)   |
| Copy             | Ctrl + Y                                 | 선택한 텍스트 복사                  |
| Go to Bottom     | Ctrl + End                               | 화면 맨 아래로 이동                 |
| Go to Top        | Ctrl + Home                              | 화면 맨 위로 이동                  |
| Insert Newline   | Alt + Enter<br>Ctrl + J<br>Shift + Enter | 프롬프트 입력창 안에서 줄바꿈            |

