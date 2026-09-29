# StoreOpsAI

점주가 매장에서 감지된 사건을 확인하고 상태를 처리하는 StoreOps AI 웹 애플리케이션입니다. React 프런트엔드와 FastAPI 백엔드를 분리해 구성했으며, Vite 개발 서버가 `/api` 요청을 백엔드로 전달합니다.

## 주요 기능

- 점주 회원가입, 로그인, 로그아웃과 세션 ID 헤더 인증
- 로그인한 점주의 매장으로 범위가 제한된 사건 목록·상세 조회
- 사건 상태 변경(확인·오탐·처리 완료)과 상태 변경 이력 조회
- `DATABASE_URL` 설정 시 PostgreSQL, 미설정 시 메모리 저장소로 자동 전환
- FastAPI 기반 헬스 체크 및 에코 API 제공
- `storeops-demand` 수요 예측·발주 초안 서비스와 세션 인증 기반 연동
- Docker Compose와 Vite 프록시를 통한 프런트엔드·백엔드 연동

## 프로젝트 구조

```text
StoreOpsAI/
├─ AGENTS.md                    # 코드 작성, 테스트, 적용 지침
├─ README.md                    # 프로젝트 문서
├─ docker-compose.yml           # 프런트엔드, 백엔드, PostgreSQL, 수요 서비스 구성
├─ docs/                        # 요구사항, DB 설계, 개발 착수 가이드
├─ backend/
│  ├─ Dockerfile               # 백엔드 컨테이너 이미지 정의
│  ├─ requirements.txt         # Python 의존성
│  ├─ migrations/              # PostgreSQL 초기 스키마
│  ├─ tests/                   # 백엔드 단위 테스트
│  └─ app/
│     ├─ main.py               # FastAPI 앱과 저장소 선택, 및 모의 사건 시드
│     ├─ database.py            # PostgreSQL 연결 생성
│     ├─ schemas/               # API 입력·출력 Pydantic 모델(auth, event)
│     ├─ repositories/          # 메모리·PostgreSQL 저장소(auth, event)
│     ├─ services/              # 인증·사건 업무 규칙
│     └─ routers/               # /api/auth, /api/events, /api/demand 라우터
├─ frontend/
│  ├─ Dockerfile                # 프런트엔드 production preview 이미지 정의
│  ├─ package.json              # npm 스크립트와 JavaScript 의존성
│  ├─ vite.config.js            # 개발 서버와 API 프록시 설정
│  ├─ index.html
│  └─ src/
│     ├─ App.jsx                # 인증 화면과 사건 관리 화면
│     ├─ App.css                # 화면 스타일
│     ├─ index.css              # 전역 스타일
│     ├─ main.jsx               # React 진입점
│     └─ api/                   # client.js, auth.js, events.js
└─ storeops_ai/
  ├─ Dockerfile                # CCTV 탐지 서비스 이미지 정의
  ├─ main.py                   # 행동분류·카메라 끊김 FastAPI 앱
  ├─ api/                      # Path 1·2 탐지 API
  └─ utils/notification.py     # 백엔드 사건 수신 웹훅 호출
```

## 요구 사항

- Windows PowerShell
- Node.js와 npm
- Python 3.10 이상 권장

## 설치

### 프런트엔드

```powershell
cd frontend
npm install
```

### 백엔드

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 실행

백엔드와 프런트엔드를 각각의 터미널에서 실행합니다.

### Docker로 백엔드와 PostgreSQL 실행

Docker Desktop을 실행한 뒤 프로젝트 루트에서 다음 명령을 실행합니다. 내부의 `storeops-demand`도 함께 빌드해 수요 서비스와 발주 화면을 연결합니다.

```powershell
docker compose up -d --build
docker compose ps
```

프런트엔드는 `http://localhost:5173`, 백엔드는 `http://localhost:8000`, 수요 서비스는 `http://localhost:8765`, PostgreSQL은 호스트의 `5433` 포트로 열립니다. 프런트엔드는 Docker 내부에서 `/api` 요청을 `backend` 컨테이너로 전달하고, 백엔드는 `/api/demand/*` 프록시를 사용하므로 수요 서비스 토큰을 브라우저에 노출하지 않습니다.

```text
Host: localhost
Port: 5433
Database: storeops
Username: storeops
Password: storeops_dev_password
```

초기 테이블은 `backend/migrations/001_initial_schema.sql`을 기반으로 PostgreSQL 최초 생성 시 자동으로 만들어집니다. 컨테이너를 중지하려면 다음 명령을 사용합니다.

```powershell
docker compose down
```

회원가입 시 서버가 새 매장 ID와 매장 기준정보를 생성합니다. 브라우저가 매장 ID를 지정하지 않으며 `STOREOPS_SINGLE_STORE_ID`와 `STOREOPS_SINGLE_STORE_NAME`은 사용하지 않습니다. Compose 실행 전 `STOREOPS_DEMAND_TOKENS_JSON`에 실제 매장별 수요 서비스 토큰을, `STOREOPS_AI_INGEST_TOKENS_JSON`에 탐지 토큰과 매장 ID의 JSON 매핑을 설정하세요.

### CCTV 탐지 서비스 연결

`storeops_ai`는 `storeops-ai` 컨테이너로 실행되며, 탐지 사건을 내부 웹훅으로 백엔드에 전달합니다. 백엔드는 인증된 탐지 토큰의 서버 설정 매핑으로 매장을 결정하고, 사건 ID를 DB에서 발급해 PostgreSQL에 저장합니다.

```text
storeops_ai -> POST /api/internal/events -> backend -> PostgreSQL -> frontend
```

Compose 전체를 실행하면 탐지 서비스 API는 `http://localhost:8100`, 내부 웹훅은 Docker 네트워크의 `http://backend:8000/api/internal/events`를 사용합니다. `EVENT_INGEST_TOKEN`은 탐지 컨테이너의 비밀값이며, 백엔드의 `STOREOPS_AI_INGEST_TOKENS_JSON` 키와 매칭되어야 합니다. 두 값 모두 브라우저에 노출하지 마세요.

실제 영상으로 확인하려면 영상 파일을 `storeops_ai/input/sample.mp4`로 복사한 뒤 다음 요청을 보냅니다.

```powershell
Copy-Item "C:\영상\sample.mp4" ".\storeops_ai\input\sample.mp4"
$body = @{ video_path = "/app/input/sample.mp4"; camera_id = "CAM-01" } | ConvertTo-Json
Invoke-RestMethod http://localhost:8100/path1/analyze -Method Post -ContentType "application/json" -Body $body
```

분석 결과에서 사건이 발급되면 영상은 `storeops_ai/output`에 저장되고, 웹훅으로 백엔드에 등록됩니다. 이후 [http://localhost:5173](http://localhost:5173)에서 로그인해 사건을 선택하면 상세 패널에서 영상을 재생할 수 있습니다. Docker 환경에서는 영상이 브라우저 호환 H.264 MP4로 변환됩니다.

DB 데이터까지 삭제하려면 다음 명령을 사용합니다. 이 명령은 개발 데이터도 삭제하므로 주의해야 합니다.

```powershell
docker compose down -v
```

### 백엔드 실행

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
$env:STOREOPS_DEMAND_URL = "http://localhost:8765"
$env:STOREOPS_DEMAND_TOKENS_JSON = '{"S01":"실제-토큰"}'
$env:STOREOPS_AI_INGEST_TOKENS_JSON = '{"실제-탐지-토큰":"S01"}'
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

백엔드 주소: http://localhost:8000

### 프런트엔드 실행

```powershell
cd frontend
npm run dev -- --host 0.0.0.0
```

프런트엔드 주소: http://localhost:5173

## 빠른 화면 시연

Docker를 사용하지 않고도 백엔드는 기본적으로 메모리 저장소로 실행되므로, 다음 순서로 화면을 바로 확인할 수 있습니다. 두 터미널을 사용합니다.

터미널 1에서 백엔드를 실행합니다.

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

터미널 2에서 프런트엔드를 실행합니다.

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

브라우저에서 [http://localhost:5173](http://localhost:5173)을 열고 다음 흐름을 따라가면 됩니다.

1. `회원가입`으로 전환한 뒤 매장 이름, 점주 이름, 이메일, 비밀번호를 입력합니다.
2. 로그인하면 가입한 매장에 접근할 수 있습니다. 신규 가입 매장은 사건이 없는 상태로 시작할 수 있습니다.
3. 사건 상세·상태 변경 화면을 시연하려면 `S01` 매장으로 인증된 계정이 필요합니다. 기본 시드 사건 `E014`(카메라 끊김), `E015`(낙상 의심)는 `S01`에 속합니다.
4. `E015`를 선택해 모델 점수와 VLM 설명 상태를 확인한 뒤 `확인 처리`, `처리 완료`를 차례로 누릅니다.
5. `E014`에서는 `오탐 처리`를 눌러 다른 상태 전이도 확인할 수 있습니다.

화면 상단에 `API 연결됨`이 표시되면 프런트엔드에서 백엔드까지 연결된 상태입니다. 직접 확인하려면 [http://localhost:8000/api/health](http://localhost:8000/api/health)를 열어 다음 응답을 확인합니다.

```json
{
  "service": "storeopsai-api",
  "status": "ok"
}
```

메모리 저장소를 사용하는 경우 백엔드를 다시 시작하면 회원가입 계정과 사건 상태 변경 내용이 초기화됩니다. PostgreSQL과 수요·발주 화면까지 함께 시연하려면 위의 Docker Compose 실행 방법을 사용합니다. 단일 매장 데모 설정에서는 회원가입 시 입력한 매장 이름 대신 `S01 데모 매장`이 사용되며 사건·발주 탭이 모두 같은 `S01` 범위를 공유합니다.

## API

### 헬스 체크

```http
GET /api/health
```

예상 응답:

```json
{
  "service": "storeopsai-api",
  "status": "ok"
}
```

### 메시지 에코

```http
POST /api/echo
Content-Type: application/json

{
	"message": "안녕하세요"
}
```

### 인증

로그인 성공 시 API가 반환한 `session_id`를 프런트엔드가 저장하고, 이후 보호된 요청에 `X-Session-ID` 헤더로 전달합니다.

| 메서드 | 경로               | 설명                             |
| ------ | ------------------ | -------------------------------- |
| POST   | `/api/auth/signup` | 매장 이름과 계정 정보로 회원가입 |
| POST   | `/api/auth/login`  | 로그인 후 세션 쿠키 발급         |
| POST   | `/api/auth/logout` | 세션 폐기와 쿠키 삭제            |
| GET    | `/api/auth/me`     | 세션 쿠키로 로그인한 점주 조회   |

### 사건

로그인한 점주의 매장 번호로 범위가 제한되며, 세션 쿠키가 없으면 `401`을 반환합니다.

| 메서드 | 경로                             | 설명                     |
| ------ | -------------------------------- | ------------------------ |
| GET    | `/api/events`                    | 사건 목록 조회           |
| GET    | `/api/events/{event_id}`         | 사건 상세 조회           |
| POST   | `/api/events/{event_id}/status`  | 사건 상태 변경           |
| GET    | `/api/events/{event_id}/history` | 사건 상태 변경 이력 조회 |

### 수요·발주

로그인 세션의 매장에 매핑된 수요 서비스 토큰으로 요청하며, 브라우저는 하위 서비스 토큰을 직접 전달하지 않습니다.
점주가 매장에서 감지된 사건을 확인하고 처리하는 웹 애플리케이션입니다. React 프런트엔드, FastAPI 업무 백엔드, CCTV 탐지 서비스, 수요 예측·발주 서비스로 구성되며 Docker Compose로 통합 실행할 수 있습니다.

## 주요 기능

- 점주 회원가입·로그인과 `X-Session-ID` 세션 헤더 인증
- 로그인한 점주의 매장으로 범위가 제한된 사건 조회, 상태 변경, 이력 확인
- 사건 영상과 대표 이미지 조회
- CCTV 행동 분석 및 카메라 연결 끊김 감시 결과를 내부 웹훅으로 수신
- `storeops-demand`의 수요 예측과 발주 초안을 인증된 백엔드 프록시로 제공
- PostgreSQL 영속 저장 또는 로컬 개발용 메모리 저장

## 구성

| 구성 요소          | 역할                                        | 기본 주소               |
| ------------------ | ------------------------------------------- | ----------------------- |
| `frontend/`        | React 19 점주 화면, Vite 개발 서버          | `http://localhost:5173` |
| `backend/`         | 인증, 사건 API, 수요 프록시, 내부 사건 수신 | `http://localhost:8000` |
| `storeops-demand/` | 수요 예측, 발주 초안 API와 데모 화면        | `http://localhost:8765` |
| `storeops_ai/`     | 영상 행동 분석 및 카메라 연결 감시 API      | `http://localhost:8100` |
| PostgreSQL         | 통합 실행 시 계정·세션·사건 저장            | `localhost:5433`        |

프런트엔드 개발 서버는 `/api` 요청을 백엔드로 프록시합니다. Compose 실행 시 프런트엔드 컨테이너도 같은 경로로 `backend`에 연결합니다. 수요 서비스 인증 토큰은 브라우저에 전달되지 않습니다.

## 저장소 구조

```text
.
├── backend/          # FastAPI 앱, 마이그레이션, 단위 테스트
├── docs/             # 요구사항, DB 설계, 개발 가이드
├── frontend/         # React/Vite 점주 웹 애플리케이션
├── storeops-demand/  # 예측·발주 서비스와 테스트
├── storeops_ai/      # CCTV 탐지 서비스와 파이프라인
├── docker-compose.yml
└── README.md
```

세부 설계와 각 서비스의 전체 설정은 [DB 설계서](docs/DB_설계서.md), [요구사항 명세서](docs/StoreOps_AI_요구사항_명세서.md), [수요 서비스 안내](storeops-demand/README.md), [CCTV 탐지 안내](storeops_ai/README.md)를 참고하세요.

## 요구 사항

- Windows PowerShell 또는 호환 터미널
- Node.js와 npm: 프런트엔드 개발용
- Python: 백엔드 및 독립 서비스 실행용. 서비스별 상세 버전과 모델 설치 조건은 각 서비스 문서 참고
- Docker Desktop 및 Docker Compose 플러그인: 전체 서비스 통합 실행용

## 빠른 시작: 화면과 백엔드

사건 API와 인증 화면을 확인하는 가장 간단한 방법입니다. 터미널을 두 개 열고 각 서비스를 실행합니다. 이 모드에서는 계정과 데이터가 메모리에 저장되며, 백엔드 재시작 시 초기화됩니다.

터미널 1에서 백엔드를 설치하고 실행합니다.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

터미널 2에서 프런트엔드를 설치하고 실행합니다.

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

브라우저에서 [http://localhost:5173](http://localhost:5173)을 열고 회원가입한 뒤 로그인합니다. 새 매장은 처음에 사건이 없습니다. 실제 탐지 사건을 보려면 CCTV 탐지 서비스와 내부 웹훅 연동을 설정하세요. 백엔드 연결은 [헬스 체크](http://localhost:8000/api/health)에서 확인할 수 있습니다.

## 전체 서비스: Docker Compose

Compose는 프런트엔드, 백엔드, PostgreSQL, 수요 서비스, CCTV 탐지 서비스를 함께 실행합니다. 백엔드는 아래 세 환경변수를 요구하므로 프로젝트 루트의 `.env`에 실제 값으로 설정한 후 기동합니다. `.env`와 토큰은 저장소에 커밋하거나 브라우저에 노출하지 마세요.

```dotenv
STOREOPS_DEMAND_TOKENS_JSON={"S01":"<수요-서비스-토큰>"}
STOREOPS_AI_INGEST_TOKENS_JSON={"<탐지-토큰>":"S01"}
STOREOPS_AI_INGEST_TOKEN=<탐지-토큰>
```

수요 토큰 맵의 키와 탐지 토큰 맵의 값은 각각 서비스를 사용할 매장 ID입니다. 신규 설치의 첫 매장은 보통 `S01`부터 발급됩니다. 가입 계정이 다른 매장 ID를 사용한다면 두 JSON 매핑도 그 ID에 맞추세요. 탐지 컨테이너의 `STOREOPS_AI_INGEST_TOKEN` 값은 백엔드 매핑의 토큰 키와 일치해야 합니다. 위 토큰 문자열은 형식 예시이며 실제 발급된 인증 토큰으로 바꿔야 합니다.

환경을 준비한 다음 프로젝트 루트에서 실행합니다.

```powershell
docker compose up -d --build
docker compose ps
```

| 서비스             | 주소                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| 점주 화면          | [http://localhost:5173](http://localhost:5173)                                                            |
| 업무 API / OpenAPI | [http://localhost:8000](http://localhost:8000) / [http://localhost:8000/docs](http://localhost:8000/docs) |
| 수요 서비스        | [http://localhost:8765](http://localhost:8765)                                                            |
| CCTV 탐지 API      | [http://localhost:8100](http://localhost:8100)                                                            |
| PostgreSQL         | `localhost:5433`                                                                                          |

PostgreSQL 접속 정보는 개발 환경 기준 `storeops` 데이터베이스, `storeops` 사용자이며 초기 비밀번호는 Compose 파일에 정의되어 있습니다. 운영 환경에서는 기본 비밀번호를 반드시 교체하세요. 테이블은 최초 DB 볼륨 생성 시 `backend/migrations/001_initial_schema.sql`로 초기화됩니다. 컨테이너를 중지하려면 `docker compose down`을 실행합니다. DB 볼륨까지 삭제하려면 `docker compose down -v`를 사용하세요. 이 명령은 저장된 개발 데이터도 삭제합니다.

## CCTV 사건 연결

탐지 서비스는 분석 결과를 `POST /api/internal/events`로 백엔드에 전달합니다. 백엔드는 `EVENT_INGEST_TOKEN`을 매장 ID에 매핑하고, 사건의 매장 범위와 ID를 서버에서 결정합니다.

```text
storeops_ai -> backend 내부 사건 API -> PostgreSQL -> 점주 화면
```

Compose 환경에서 분석할 영상 파일을 `storeops_ai/input/sample.mp4`에 둔 뒤 다음 요청을 보냅니다.

```powershell
$body = @{ video_path = "/app/input/sample.mp4"; camera_id = "CAM-01" } | ConvertTo-Json
Invoke-RestMethod http://localhost:8100/path1/analyze -Method Post -ContentType "application/json" -Body $body
```

사건이 생성되면 탐지 결과와 미디어가 저장되고, 웹훅을 통해 백엔드에 등록됩니다. 매장 계정으로 화면에 로그인해 사건을 확인하세요. 영상 모델·임계값·처리 파이프라인은 [CCTV 탐지 서비스 안내](storeops_ai/README.md)를 참고하세요.

## API 개요

로그인 후 보호 API를 호출할 때 로그인 응답의 `session_id`를 `X-Session-ID` 헤더로 전달합니다. 사건과 발주 데이터는 세션 점주의 매장으로 제한됩니다.

| 메서드        | 경로                                           | 용도                    |
| ------------- | ---------------------------------------------- | ----------------------- |
| `GET`         | `/api/health`                                  | 백엔드 상태 확인        |
| `POST`        | `/api/echo`                                    | 연결 확인용 메시지 에코 |
| `POST`        | `/api/auth/signup`                             | 매장 및 점주 계정 생성  |
| `POST`        | `/api/auth/login`                              | 로그인 및 세션 발급     |
| `POST`        | `/api/auth/logout`                             | 세션 폐기               |
| `GET`         | `/api/auth/me`                                 | 현재 로그인 사용자 조회 |
| `GET`         | `/api/events`                                  | 사건 목록 조회          |
| `GET`         | `/api/events/{event_id}`                       | 사건 상세 조회          |
| `POST`        | `/api/events/{event_id}/status`                | 사건 상태 변경          |
| `GET`         | `/api/events/{event_id}/history`               | 상태 변경 이력 조회     |
| `GET`, `POST` | `/api/demand/orders/drafts`                    | 발주 초안 조회·생성     |
| `POST`        | `/api/demand/orders/drafts/{draft_id}/approve` | 발주 초안 승인          |
| `POST`        | `/api/internal/events`                         | 인증된 탐지 사건 수신   |

전체 요청·응답 스키마는 백엔드의 `/docs`에서 확인할 수 있습니다. 발주 초안 생성 요청은 `Idempotency-Key` 헤더를 사용합니다.

## 검증

백엔드 단위 테스트와 문법 검사를 실행합니다.

```powershell
cd backend
python -m compileall app
python -m unittest discover -s tests -p "test_*.py" -v
```

프런트엔드 정적 검사와 프로덕션 빌드를 실행합니다.

```powershell
cd frontend
npm install
npm run lint
npm run build
```

수요 서비스 테스트는 해당 폴더의 가상환경에 의존성을 설치한 뒤 실행합니다.

```powershell
cd storeops-demand
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install --no-deps -e .
python -m pytest -q
```

CCTV 탐지 서비스의 설치, 단독 실행, 영상 테스트는 [서비스 README](storeops_ai/README.md)에 설명되어 있습니다. 서비스별 실행 및 검증 방법은 [개발 지침](AGENTS.md)도 참고하세요.

## 라이선스

현재 저장소에는 별도 라이선스가 지정되어 있지 않습니다.
