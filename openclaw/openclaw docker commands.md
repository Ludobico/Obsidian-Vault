
| **목적**         | **일반 로컬 명령어**                    | **Docker Compose 명령어**                                                                            |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------------- |
| 초기 설정 마법사 실행   | `openclaw onboard`               | `docker compose run --rm openclaw-cli onboard`                                                    |
| 게이트웨이(서버) 시작   | `openclaw gateway start`         | `docker compose up -d openclaw-gateway`                                                           |
| 게이트웨이(서버) 종료   | `openclaw gateway stop`          | `docker compose stop openclaw-gateway`<br><br>(또는 `docker compose down`)                          |
| 게이트웨이 재시작      | `openclaw gateway restart`       | `docker compose restart openclaw-gateway`                                                         |
| 대시보드 접속 URL 확인 | `openclaw dashboard --no-open`   | `docker compose run --rm openclaw-cli dashboard --no-open`                                        |
| 페어링 요청 기기 목록   | `openclaw devices list`          | `docker compose run --rm openclaw-cli devices list`                                               |
| 특정 기기 페어링 승인   | `openclaw devices approve [ID]`  | `docker compose run --rm openclaw-cli devices approve [ID]`                                       |
| 설정 상태 및 에러 진단  | `openclaw doctor`                | `docker compose run --rm openclaw-cli doctor`                                                     |
| 현재 게이트웨이 상태    | `openclaw status`                | `docker compose run --rm openclaw-cli status`                                                     |
| 기존 토큰값 확인하기    | `openclaw config get gateway...` | `docker compose run --rm openclaw-cli config get gateway.auth.token`                              |
| 실시간 로그 보기      | (별도 명령어 없음)                      | `docker compose logs -f openclaw-gateway`                                                         |
| 텔레그램 코드 페어링    |                                  | `docker compose exec openclaw-gateway node dist/index.js pairing approve telegram YOUR_CODE_HERE` |
