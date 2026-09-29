<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI">
</p>

# StoreOps AI 백엔드

StoreOps AI 웹 애플리케이션의 FastAPI 백엔드입니다. 점주 인증과 세션을 관리하고, 점주의 매장으로 범위가 제한된 사건 API를 제공합니다. 수요·발주 서비스로 요청을 전달하고, CCTV 탐지 서비스에서 발생한 사건을 내부 API로 수신합니다.

## 주요 기능

- 점주 회원가입·로그인·로그아웃 및 `X-Session-ID` 세션 인증
- 매장별 사건 목록, 상세, 상태 변경, 이력 및 미디어 조회
- 탐지 토큰과 매장 매핑을 이용한 내부 사건 수신
- 매장별 인증 토큰을 사용하는 수요·발주 서비스 프록시
- `DATABASE_URL` 설정에 따른 PostgreSQL 또는 메모리 저장소 선택

## 요구 사항

- Python 3.12 권장. Docker 이미지는 Python 3.12를 사용합니다.
- Windows PowerShell 또는 호환 터미널
- PostgreSQL은 영속 저장에만 필요합니다. 로컬 메모리 모드로도 API를 실행할 수 있습니다.

프로젝트 전체 구성과 Compose 환경변수는 [루트 README](../README.md)를 참고하세요. 데이터베이스 구조는 [`migrations/001_initial_schema.sql`](migrations/001_initial_schema.sql)에 정의되어 있습니다.

## 설치 및 로컬 실행

프로젝트 루트 기준으로 백엔드 가상환경을 만들고 개발 서버를 실행합니다.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

기본 실행은 `DATABASE_URL`이 설정되지 않은 메모리 저장소 모드입니다. 회원 계정, 세션, 사건은 프로세스 메모리에만 보관되며 백엔드가 재시작되면 초기화됩니다. 서버는 [http://localhost:8000](http://localhost:8000), OpenAPI 문서는 [http://localhost:8000/docs](http://localhost:8000/docs)에서 확인합니다. 상태 확인 주소는 [http://localhost:8000/api/health](http://localhost:8000/api/health)입니다.

## 환경 설정

| 변수                             | 기본값                  | 설명                                                             |
| -------------------------------- | ----------------------- | ---------------------------------------------------------------- |
| `DATABASE_URL`                   | 미설정                  | PostgreSQL 연결 문자열. 설정하면 PostgreSQL 저장소를 사용합니다. |
| `FRONTEND_ORIGIN`                | `http://localhost:5173` | CORS 요청을 허용할 프런트엔드 출처입니다.                        |
| `STOREOPS_OUTPUT_DIR`            | `/app/output`           | 사건 영상·대표 이미지 파일을 제공할 출력 폴더입니다.             |
| `STOREOPS_DEMAND_URL`            | `http://localhost:8765` | 수요 서비스 기본 주소입니다.                                     |
| `STOREOPS_DEMAND_TOKENS_JSON`    | `{}`                    | 매장 ID를 수요 서비스 토큰에 대응시키는 JSON 객체입니다.         |
| `STOREOPS_DEMAND_TOKEN`          | 미설정                  | 매장별 토큰 맵에 항목이 없을 때 사용하는 선택적 대체 토큰입니다. |
| `STOREOPS_AI_INGEST_TOKENS_JSON` | 미설정                  | CCTV 탐지 토큰을 매장 ID에 대응시키는 JSON 객체입니다.           |

Compose 환경에서 사용하는 예시는 [`backend/.env.example`](.env.example)에 있습니다. 해당 예시는 컨테이너 내부의 PostgreSQL 호스트명 `postgres`를 사용하므로, 백엔드를 Windows에서 직접 실행하면서 PostgreSQL에 연결할 때는 `localhost:5433` 주소로 된 연결 문자열을 사용해야 합니다. `.env` 파일이나 실제 토큰은 저장소에 커밋하지 마세요.

PowerShell에서 수요 서비스와 내부 사건 수신 설정을 지정하는 예시입니다. 토큰과 매장 ID는 실제 환경과 일치하도록 바꾸세요.

```powershell
$env:STOREOPS_DEMAND_URL = "http://localhost:8765"
$env:STOREOPS_DEMAND_TOKENS_JSON = '{"S01":"<수요-서비스-토큰>"}'
$env:STOREOPS_AI_INGEST_TOKENS_JSON = '{"<탐지-토큰>":"S01"}'
```

## PostgreSQL 모드

`DATABASE_URL`을 설정하면 계정, 세션, 사건 저장소가 PostgreSQL을 사용합니다. 스키마는 [`migrations/001_initial_schema.sql`](migrations/001_initial_schema.sql)에 있습니다. Compose에서는 PostgreSQL 컨테이너 최초 생성 시 이 파일이 자동 적용됩니다. 기존 데이터베이스에 직접 연결하는 경우에는 마이그레이션을 별도로 적용해야 합니다.

```powershell
$env:DATABASE_URL = "postgresql://storeops:<비밀번호>@localhost:5433/storeops"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API 개요

로그인 응답의 `session_id`를 보호된 요청의 `X-Session-ID` 헤더에 담아 보냅니다. 보호 API는 점주의 세션과 연결된 매장 기준으로 데이터를 제한합니다. 로그인 세션의 유효 기간은 7일이며 로그아웃하면 폐기됩니다.

| 메서드        | 경로                                                        | 인증        | 설명                   |
| ------------- | ----------------------------------------------------------- | ----------- | ---------------------- |
| `GET`         | `/api/health`                                               | 없음        | 백엔드 상태 확인       |
| `POST`        | `/api/echo`                                                 | 없음        | 요청 메시지 에코       |
| `POST`        | `/api/auth/signup`                                          | 없음        | 매장과 점주 계정 생성  |
| `POST`        | `/api/auth/login`                                           | 없음        | 점주 인증 및 세션 발급 |
| `POST`        | `/api/auth/logout`                                          | 세션 헤더   | 세션 폐기              |
| `GET`         | `/api/auth/me`                                              | 세션 헤더   | 로그인 사용자 조회     |
| `GET`         | `/api/events`                                               | 세션 헤더   | 매장 사건 목록 조회    |
| `GET`         | `/api/events/{event_id}`                                    | 세션 헤더   | 사건 상세 조회         |
| `POST`        | `/api/events/{event_id}/status`                             | 세션 헤더   | 사건 상태 변경         |
| `GET`         | `/api/events/{event_id}/history`                            | 세션 헤더   | 상태 변경 이력 조회    |
| `GET`         | `/api/events/{event_id}/clip`                               | 세션 헤더   | 사건 영상 조회         |
| `GET`         | `/api/events/{event_id}/representative-images/{media_type}` | 세션 헤더   | 대표 이미지 조회       |
| `GET`, `POST` | `/api/demand/orders/drafts`                                 | 세션 헤더   | 발주 초안 조회 및 생성 |
| `POST`        | `/api/demand/orders/drafts/{draft_id}/approve`              | 세션 헤더   | 발주 초안 승인         |
| `POST`        | `/api/internal/events`                                      | Bearer 토큰 | CCTV 탐지 사건 수신    |

상태 값은 `unconfirmed`, `confirmed`, `resolved`, `false_alarm`입니다. 발주 초안 생성에는 `Idempotency-Key` 헤더가 필요합니다. 요청·응답 스키마와 전체 명세는 실행 중 `/docs`에서 확인할 수 있습니다.

## 서비스 연동

### 수요·발주 서비스

백엔드는 `/api/demand/*` 요청을 `STOREOPS_DEMAND_URL`로 전달합니다. 사용자 세션에서 매장을 확인한 뒤 `STOREOPS_DEMAND_TOKENS_JSON`에서 해당 매장의 서비스 토큰을 찾아 하위 서비스의 Bearer 인증에 사용합니다. 따라서 수요 서비스 토큰을 브라우저에 제공하지 않습니다.

### CCTV 탐지 서비스

탐지 서비스가 `POST /api/internal/events`로 사건을 보낼 때 Bearer 토큰을 확인하고, `STOREOPS_AI_INGEST_TOKENS_JSON`의 서버 측 매핑으로 매장을 결정합니다. 요청 본문으로 전달된 매장 ID는 권한 판정에 사용하지 않습니다. 사건 영상과 이미지를 브라우저에서 조회하려면 백엔드와 탐지 서비스가 같은 출력 파일을 볼 수 있도록 공유 볼륨과 `STOREOPS_OUTPUT_DIR`을 맞춰야 합니다.

## 테스트 및 검증

가상환경을 활성화한 백엔드 폴더에서 실행합니다.

```powershell
python -m compileall app
python -m unittest discover -s tests -p "test_*.py" -v
```

## Docker Compose

프로젝트 루트의 `.env`에 수요 서비스 토큰 맵, 탐지 토큰 맵, 탐지 컨테이너 토큰을 설정한 뒤 프로젝트 루트에서 실행합니다.

```powershell
docker compose up -d --build backend postgres demand storeops-ai
docker compose ps
```

프런트엔드까지 포함한 전체 서비스 실행 방법과 필수 환경변수 예시는 [루트 README](../README.md)를 참고하세요.

## 기여

변경 후 백엔드 문법 검사와 관련 단위 테스트를 실행합니다. 변경 사항이 PostgreSQL 스키마에 영향을 주면 마이그레이션과 관련 테스트도 함께 갱신합니다. 문서와 설명은 프로젝트 지침에 따라 한국어로 작성합니다.

## 라이선스

현재 저장소에는 별도 라이선스가 지정되어 있지 않습니다.
