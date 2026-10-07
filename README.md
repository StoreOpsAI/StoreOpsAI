<p align="center">
  <img src="https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white" alt="React 19">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker Compose">
</p>

# StoreOpsAI

점주가 CCTV 탐지 사건을 확인하고 처리하며, 매장 수요 예측을 바탕으로 발주 초안을 관리하고 운영 기록과 점검 규정을 질문할 수 있는 프로젝트입니다. React 프런트엔드, FastAPI 백엔드, CCTV 탐지 서비스, 수요·발주 서비스와 PostgreSQL로 구성되며, Q&A·LLM 서버는 선택적으로 별도 실행합니다.

## 주요 기능

- 점주 회원가입·로그인 및 `X-Session-ID` 세션 인증
- 점주 매장으로 범위가 제한된 사건 조회, 상세·미디어 확인, 상태 변경 및 이력 조회
- CCTV 행동 분석과 카메라 연결 끊김 사건 수신
- 수요 예측 결과 기반 발주 초안 생성, 조회와 승인
- 로그인한 점주의 사건·발주 기록과 매장 점검 규정 질문 및 근거 확인
- PostgreSQL 영속 저장 및 로컬 개발용 메모리 저장소

## 저장소 구조

아래는 현재 저장소의 주요 파일과 서비스별 디렉터리입니다. 가상환경, 캐시, 모델 가중치와 실행 중 생성되는 데이터는 제외했습니다.

```text
StoreOpsAI/
├── AGENTS.md                         # 작업 지침
├── README.md                         # 프로젝트 및 개발환경 안내
├── docker-compose.yml                # 메인 PC용 6개 컨테이너
├── llama.cpp/llama.cpp/              # Windows CPU·Vulkan·CUDA 실행 파일
├── backend/                          # FastAPI 업무 API
│   ├── app/
│   │   ├── routers/                  # auth, demand, events, internal_events, questions
│   │   ├── repositories/             # 인증·사건 메모리/PostgreSQL 저장소
│   │   ├── schemas/                  # API 모델
│   │   └── services/                 # 인증·사건 업무 로직
│   ├── migrations/                   # PostgreSQL 스키마 변경
│   ├── sql/                          # SQL 스크립트
│   └── tests/                        # 백엔드 테스트
├── frontend/                         # React 19 + Vite 점주 화면
│   └── src/
│       ├── api/                      # auth, demand, events, questions API
│       └── assets/                   # 화면 리소스
├── storeops-demand/                  # 수요 예측·발주 초안
│   ├── storeops/                     # API, 데이터, 예측, 주문, 학습
│   ├── artifacts/                    # 모델·평가 결과
│   ├── examples/                     # 예제 데이터
│   ├── reports/                      # 검증 보고서
│   ├── runtime/                      # SQLite 실행 데이터
│   └── tests/
└── storeops_ai/                      # CCTV 분석·카메라 감시
  ├── api/                          # 영상 분석·카메라 연결 API
  ├── config/                       # 탐지·모델 설정
  ├── models/                       # YOLO, S3D, VLM
  ├── pipelines/                    # 행동 분석·웹캠·연결 감시
  ├── input/                        # 입력·실패·처리 영상
  ├── output/                       # 사건 클립·이미지·JSON
  ├── tools/                        # 모델 확인·Qwen 서버 실행
  └── training/                     # manifests, runs, scripts
```

세부 설정과 사용법은 [백엔드 README](backend/README.md), [프런트엔드 README](frontend/README.md), [수요 서비스 README](storeops-demand/README.md), [CCTV 탐지 README](storeops_ai/README.md)를 참고하세요. 질문 Agent는 저장소 외부 PC에서 운영하며, 연결 값은 루트 `.env`에 설정합니다.

## 구성 및 주소

| 서비스           | Compose 서비스 | 로컬 주소                                                | 용도                          |
| ---------------- | -------------- | -------------------------------------------------------- | ----------------------------- |
| 점주 프런트엔드  | `frontend`     | [http://localhost:5173](http://localhost:5173)           | 사건·발주 화면                |
| 업무 백엔드      | `backend`      | [http://localhost:8000](http://localhost:8000)           | 인증, 사건 API, 서비스 프록시 |
| 백엔드 API 문서  | `backend`      | [http://localhost:8000/docs](http://localhost:8000/docs) | OpenAPI/Swagger               |
| 수요·발주 서비스 | `demand`       | [http://localhost:8765](http://localhost:8765)           | 예측·발주 API 및 데모         |
| CCTV 탐지 서비스 | `storeops-ai`  | [http://localhost:8100](http://localhost:8100)           | 영상 분석 API                 |
| PostgreSQL       | `postgres`     | `localhost:5433`                                         | 업무 데이터 영속 저장         |
| Qwen VLM 서버    | `qwen-vllm`    | `http://localhost:8001`                                  | CCTV 대표 이미지 설명         |
| 질문 Agent       | 원격 Docker     | `http://<QNA_PC_IP>:8002`                                | 질문 처리 및 규정 검색        |

프런트엔드 개발 서버와 Nginx는 `/api`를 백엔드로, `/detector`를 메인 PC의 CCTV 탐지 서비스로 프록시합니다. 감시용 Qwen VLM은 메인 PC Compose에서 실행하지만 질문 Agent와 텍스트 LLM은 별도 PC에서 운영합니다. 브라우저는 백엔드만 호출하고 수요·질문 서비스 인증 토큰은 백엔드가 보관합니다. 원격 Agent 주소는 `STOREOPS_QNA_URL`, 공유 토큰은 `STOREOPS_QNA_TOKEN`으로 설정합니다.

## 개발환경 가이드

Windows 10/11 x64와 PowerShell 기준입니다. AI 기능마다 런타임이 다르므로 Qwen용 WSL 가상환경을 기본 Python 환경과 분리하세요.

| 개발 도구       | 버전                               | 사용처                                    |
| --------------- | ---------------------------------- | ----------------------------------------- |
| Git             | 최신 안정 버전                     | 저장소 내려받기                           |
| Python          | 3.12.x                             | 백엔드, 수요 서비스, CCTV 분석, Q&A Agent |
| Node.js 및 npm  | Node.js 22.x LTS 및 포함된 npm     | React 프런트엔드                          |
| Docker Desktop  | 최신 안정 버전, Compose v2 포함    | 전체 서비스 실행                          |
| WSL2 + Ubuntu   | Ubuntu 26.04.1 LTS                 | Qwen 비전 모델 서버 전용                  |
| NVIDIA 드라이버 | CUDA 13.0을 지원하는 최신 드라이버 | WSL2 Qwen 비전 모델 GPU 실행              |

설치 후 PowerShell에서 도구를 확인합니다. Docker와 NVIDIA/WSL 명령은 해당 AI 경로를 사용할 때 확인하면 됩니다.

```powershell
git --version
py -3.12 --version
node --version
npm --version
docker compose version
wsl --status
nvidia-smi
```

### 처음 실행하기

1. 화면과 API만 개발하려면 [프런트엔드와 백엔드 로컬 실행](#프런트엔드와-백엔드만-로컬-실행) 절차를 따릅니다. 두 터미널을 사용하며, 백엔드는 PostgreSQL 없이 메모리 저장소로 실행됩니다.
2. 메인 PC에서 PostgreSQL·백엔드·프런트엔드·수요·CCTV 탐지·Qwen VLM을 실행하려면 [전체 Docker 실행](#전체-서비스-실행)을 선택합니다. 질문 Agent와 텍스트 LLM은 별도 PC에서 먼저 실행되어야 합니다. 감시용 Qwen VLM에는 NVIDIA GPU 환경이 필요합니다.
3. 사용할 AI 기능을 아래 표에서 골라 필요한 모델과 추가 환경을 준비합니다.

백엔드는 `load_dotenv()`로 루트 `.env`를 읽을 수 있습니다. Compose 내부 주소(`postgres`, `demand`)는 메인 PC Docker 네트워크에서 사용하고, `STOREOPS_QNA_URL`에는 원격 Q&A PC의 LAN 주소를 지정합니다. 호스트에서 백엔드를 직접 실행할 때는 각 서비스 주소를 해당 실행 환경에 맞게 설정하고, 메모리 저장소를 원하면 `DATABASE_URL`을 빈 값으로 설정하세요. 실제 토큰은 저장소에 커밋하지 마세요.

### AI 기능별 환경

| 기능                       | 버전 및 하드웨어                                                                                                  | 모델 준비 및 실행                                                                                                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CCTV 행동 탐지             | Python 3.12 권장, PyTorch 2.0 이상, torchvision 0.15 이상. CPU 가능, NVIDIA GPU는 선택                            | YOLO 및 S3D 가중치가 저장소에 포함됩니다. 아래 CCTV 환경을 준비하고 `python -m tools.verify_i3d`로 확인합니다.                                                                 |
| 수요 예측                  | Python 3.12, pandas 3.0.6, xgboost-cpu 3.4.1, scikit-learn 1.9.1. GPU 불필요                                      | 학습 모델이 포함되어 있습니다. 단독 실행은 [수요 서비스 README](storeops-demand/README.md), 전체 연동은 Docker Compose를 사용합니다.                                           |
| 점주 질문 서비스           | 원격 Agent·LLM PC                                                                                                   | 질문 Agent와 텍스트 LLM은 이 저장소의 Compose에 포함되지 않습니다. 원격 서비스 주소와 공유 토큰을 `.env`에 설정합니다.                                                    |
| CCTV 이미지 설명(Qwen VLM) | WSL2 Ubuntu 26.04.1, Python 3.14, PyTorch 2.13.0+cu130, CUDA 13.0, FlashInfer 0.6.18.post1, vLLM. NVIDIA GPU 필요 | Qwen3-VL-8B-Instruct-FP8 모델을 내려받습니다. RTX A4000 16GB에서 검증된 구성입니다. 전체 설치 절차는 [CCTV 탐지 README](storeops_ai/README.md)의 WSL2/vLLM 항목을 따릅니다.    |

Compose의 Qwen VLM은 `vllm/vllm-openai` 컨테이너로 실행됩니다. 직접 WSL에서 실행하는 경우에는 CCTV 탐지 README의 검증된 Python 3.14 환경을 따르고, 다른 서비스의 Python 3.12 가상환경에 vLLM을 설치하지 마세요. 원격 질문 Agent의 GGUF와 매뉴얼 임베딩 모델은 해당 PC에서 별도로 준비해야 하며, Compose의 Qwen VLM 가중치는 첫 기동 시 Docker 볼륨에 내려받습니다.

먼저 CCTV 행동 모델만 로컬에서 확인하려면:

```powershell
cd storeops_ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m tools.verify_i3d
```

서비스를 Compose와 분리해 로컬로 직접 실행할 때는 VLM 없이도 CCTV 행동 탐지와 수요 예측을 사용할 수 있습니다. 다만 현재 Compose의 `storeops-ai` 서비스는 `qwen-vllm`의 healthy 상태를 기다리므로 전체 Compose 실행에는 Qwen VLM용 GPU 환경이 필요합니다. Qwen 기반 질문 Agent와 이미지 설명은 각각 해당 가이드의 모델 준비를 마친 뒤 사용하세요.

## Docker 구성 확인

Compose에는 메인 PC의 프런트엔드, 백엔드, 수요 서비스, CCTV 탐지기, PostgreSQL, 감시용 Qwen VLM이 정의되어 있습니다. 질문 Agent와 텍스트 LLM은 원격 PC에서 별도로 실행합니다. 실행 후 아래 명령으로 메인 PC 서비스를 확인하세요.

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/api/health
Invoke-RestMethod http://localhost:8765/health
Invoke-RestMethod http://localhost:8100/
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod http://<QNA_PC_IP>:8002/openapi.json
```

## 실행 방법

### 전체 서비스 실행

Docker Desktop과 Docker Compose가 필요합니다. 프로젝트 루트의 `.env`에 Compose 필수 토큰을 설정하세요. 실제 토큰을 저장소에 추가하거나 브라우저에 노출하지 마세요.

```dotenv
STOREOPS_DEMAND_TOKENS_JSON={"S01":"demo-owner-s01"}
STOREOPS_AI_INGEST_TOKENS_JSON={"<탐지-토큰>":"S01"}
STOREOPS_AI_INGEST_TOKEN=<탐지-토큰>
STOREOPS_QNA_TOKEN=<질문-서비스-공유-토큰>
STOREOPS_QNA_URL=http://<QNA_PC_IP>:8002
```

수요 컨테이너는 현재 `STOREOPS_DEMO=1`로 실행되므로 데모 토큰은 `demo-owner-s01`로 고정되어 있습니다. 백엔드의 수요 토큰 맵은 매장 ID를 이 토큰에 연결합니다. 탐지 토큰 맵은 탐지 토큰을 매장 ID에 연결하며, `STOREOPS_AI_INGEST_TOKEN`은 탐지 컨테이너가 웹훅에 사용하는 토큰으로 맵의 토큰 키와 같아야 합니다. `STOREOPS_QNA_TOKEN`은 메인 백엔드와 원격 질문 Agent가 공유하는 필수 토큰입니다. 질문 모델 파일은 원격 Q&A PC에서 관리하며 메인 PC Compose에는 필요하지 않습니다. 새 데이터베이스에서 최초 생성되는 매장 ID는 보통 `S01`이지만, 기존 DB를 사용할 때에는 실제 가입 계정의 매장 ID에 맞추세요. 이 데모 토큰 설정을 운영망에 노출하지 마세요.

Compose의 Qwen VLM에는 Docker Desktop의 WSL2 NVIDIA GPU 지원이 필요하며, 첫 실행에서 모델을 Docker 볼륨에 내려받습니다. 원격 질문 Agent와 텍스트 LLM은 이 저장소의 Compose 빌드 대상이 아니므로 별도 PC에서 준비해야 합니다.

```powershell
docker compose up -d --build
docker compose ps
```

컨테이너 로그는 서비스별로 확인할 수 있습니다.

```powershell
docker compose logs -f backend
docker compose logs -f storeops-ai
docker compose logs -f qwen-vllm
```

메인 Compose 네트워크에서 감시 서비스는 `qwen-vllm`을 사용합니다. 원격 Q&A Agent의 LAN 주소와 `STOREOPS_QNA_TOKEN`은 `.env`에서 설정합니다. 사건 미디어는 메인 PC에서 백엔드와 감시기가 공유하는 `storeops_ai/output` 폴더에 저장됩니다.

일반 종료는 `docker compose down`입니다. DB 볼륨까지 삭제하는 `docker compose down -v`는 저장된 개발 데이터도 지우므로 주의하세요.

Compose의 PostgreSQL 사용자·DB·비밀번호는 개발용 값으로 설정되어 있습니다. 운영 환경에서는 반드시 별도 비밀값으로 교체하고 네트워크 노출을 제한하세요.

### 프런트엔드와 백엔드만 로컬 실행

이 모드는 PostgreSQL과 탐지·수요 서비스를 요구하지 않습니다. 백엔드는 `DATABASE_URL`이 없으면 메모리 저장소를 사용하므로 프로세스를 재시작하면 계정, 세션, 사건 상태가 초기화됩니다. 두 터미널에서 각각 실행합니다.

백엔드 터미널:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:DATABASE_URL = ""
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

프런트엔드 터미널:

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

브라우저에서 [http://localhost:5173](http://localhost:5173)을 열고 회원가입 후 로그인합니다. 신규 매장은 시드 사건 없이 시작합니다. 사건 목록을 확인하려면 탐지 서비스에서 사건을 생성해 백엔드로 전송해야 합니다.

## 사건 수신 흐름

CCTV 탐지 서비스가 분석 결과를 내부 사건 API에 보내면 백엔드는 토큰-매장 매핑을 확인하고 사건 ID와 매장을 결정해 PostgreSQL에 저장합니다. 탐지 서비스와 백엔드는 같은 출력 폴더를 공유합니다.

```text
storeops_ai -> POST /api/internal/events -> backend -> PostgreSQL -> frontend
```

전체 Compose 실행 후 입력 영상을 `storeops_ai/input/sample.mp4`에 두고 분석 요청을 보낼 수 있습니다.

```powershell
$body = @{ video_path = "/app/input/sample.mp4"; camera_id = "CAM-01" } | ConvertTo-Json
Invoke-RestMethod http://localhost:8100/path1/analyze -Method Post -ContentType "application/json" -Body $body
```

탐지 파이프라인과 모델 설정은 [CCTV 탐지 README](storeops_ai/README.md)를 참고하세요.

## API 개요

로그인 응답의 `session_id`를 보호된 요청의 `X-Session-ID` 헤더로 전달합니다. 사건과 발주 API는 세션 점주의 매장으로 범위가 제한됩니다.

| 메서드        | 경로                                                        | 설명                            |
| ------------- | ----------------------------------------------------------- | ------------------------------- |
| `GET`         | `/api/health`                                               | 백엔드 상태 확인                |
| `POST`        | `/api/auth/signup`                                          | 매장·점주 계정 생성             |
| `POST`        | `/api/auth/login`                                           | 로그인 및 세션 발급             |
| `POST`        | `/api/auth/logout`                                          | 세션 폐기                       |
| `GET`         | `/api/auth/me`                                              | 현재 로그인 사용자 조회         |
| `GET`         | `/api/events`                                               | 매장 사건 목록                  |
| `GET`         | `/api/events/{event_id}`                                    | 사건 상세                       |
| `POST`        | `/api/events/{event_id}/status`                             | 사건 상태 변경                  |
| `GET`         | `/api/events/{event_id}/history`                            | 상태 변경 이력                  |
| `GET`         | `/api/events/{event_id}/clip`                               | 사건 영상                       |
| `GET`         | `/api/events/{event_id}/representative-images/{media_type}` | 대표 이미지                     |
| `GET`, `POST` | `/api/demand/orders/drafts`                                 | 발주 초안 조회·생성             |
| `POST`        | `/api/demand/orders/forecast-draft`                         | 수요 예측 기반 발주 초안 생성   |
| `POST`        | `/api/demand/orders/drafts/{draft_id}/approve`              | 발주 초안 승인                  |
| `POST`        | `/api/internal/events`                                      | Bearer 토큰 인증 탐지 사건 수신 |
| `POST`        | `/api/ask`                                                  | 로그인 점주의 운영 기록 질문    |
| `GET`         | `/api/ask/{question_session_id}/log`                        | 질문 처리 도구 실행 로그 조회   |

발주 초안 생성은 `Idempotency-Key` 헤더를 사용합니다. 전체 요청·응답 형식은 백엔드의 `/docs`에서 확인할 수 있습니다.

## 테스트 및 빌드

백엔드:

```powershell
cd backend
python -m compileall app
python -m unittest discover -s tests -p "test_*.py" -v
```

프런트엔드:

```powershell
cd frontend
npm install
npm run lint
npm run build
```

수요 서비스(가상환경에서 실행):

```powershell
cd storeops-demand
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install --no-deps -e .
python -m pytest -q
```

## 기여 및 라이선스

작업 규칙, 브랜치·커밋 관례, 서비스별 검증 절차는 [AGENTS.md](AGENTS.md)를 참고하세요. 변경 후 해당 서비스의 lint·build 또는 테스트를 실행합니다.

현재 저장소에는 별도 라이선스가 지정되어 있지 않습니다.
