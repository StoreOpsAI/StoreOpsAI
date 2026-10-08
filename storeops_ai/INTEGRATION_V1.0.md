# StoreOps AI 영상 모델 v1.0 연동 안내

기존 단일 S3D 분류기 대신 **화면 전체(A) + 사람 크롭(C1) S3D 두 개를 동적 결합**한 분류기를 `storeops_ai/`에 연결하고,
VLM은 **사건이 난 순간이 아니라 점주가 지난 사건을 물을 때만** 쓰도록 구조를 바꿨습니다.

## 전체 흐름

```text
[경보]  카메라 → 탐지 서비스(YOLO + A/C1 S3D) → 사건 발생 시 백엔드 /api/internal/events 로 1회 전송
        전송 내용: 사건 종류·시각·카메라 + 클립 + 대표 이미지 3장   (점수·임계값은 보내지 않음)
        → 점주 화면에 "쓰러짐 의심" 같은 경보와 함께 해당 사건 영상·이미지를 바로 제공. VLM 칸은 "해당 없음"

[질문]  점주 "오늘 무슨 일 있었어?" → 백엔드 /api/ask → 질문 Agent(LLM)
        ① query_records : 그날 사건 번호·종류·시각·카메라 목록
        ② describe_events: 그 번호의 대표 이미지를 백엔드가 VLM(Qwen)에 보내 장면 설명을 받음
        → 사건별 설명 + "자동 해석이라 틀릴 수 있음, 영상으로 확인" 안내
```

## 서비스 카테고리 (v1.0)

| 서비스 점수 | 학습 라벨 | 백엔드 event_type |
| --- | --- | --- |
| 정상 | normal | - |
| 쓰러짐 | fall | fall |
| 쓰레기 투기 | abandon | littering |
| 절도 | theft | theft |

싸움·파손·방화는 v1.0에서 제외합니다(학습하지 않음). 6클래스(파손·방화 포함) 학습은 v1.1에서 다룹니다.

## 변경 파일

**탐지 서비스 (`storeops_ai/`)**

| 파일 | 내용 |
| --- | --- |
| `config/config.py` | 카테고리 4개, C1 가중치·결합 파일 경로, 임계값을 `fusion.json`에서 읽기, `VLM_ENABLED`, `CROP_LARGEST_PERSON`, **`VLM_ON_EVENT`(기본 0)** |
| `models/stack_head.py` (신규) | A+C1 동적 결합 `z = (a\|a\| + b\|b\|) / (\|a\| + \|b\|)` (학습 파라미터 없음), 사람 박스 합집합 크롭 |
| `models/i3d_classifier.py` | C1 로드, `predict_pair()` |
| `pipelines/path1_behavior.py` | 사람이 없는 창도 분류, C1 크롭 입력, 사건 시점 VLM 호출 제거(`VLM_ON_EVENT=0`) |
| `pipelines/path1_webcam.py` | 실시간 웹캠 모드: A+C1 결합, 사람 게이트 제거, 확인 횟수 기본 1, 사건 시점 VLM 호출 제거 |
| `utils/notification.py` | 경보 본문에서 `scores`·`threshold`·`confidence`·`risk_level`·`score_logit` 제외(탐지 서비스의 사건 JSON 파일에는 전체 값 보존) |
| `schemas/event_schema.py` | 사건 시각을 시간대 포함으로, 저장 폴더 기준 상대 경로로 정규화, 내부 필드 추가 |
| `requirements.txt`, `webcam_i3d.py`, `README.md`, `WEBCAM_TEST.md`, `tests/` | 필드 추가, WebSocket 의존성, 4카테고리 기준 정리, 테스트 |

**백엔드 (`backend/`)**

| 파일 | 내용 |
| --- | --- |
| `app/schemas/event.py` | VLM 결과 입력에 `not_applicable` 허용(점주 화면에 "해당 없음"). DB의 `vlm_results.status` 제약은 이미 허용하므로 마이그레이션 불필요 |
| `app/routers/internal_qna.py` (신규) | `POST /api/internal/qna/describe`: 질문 Agent가 사건 번호를 주면 요청한 **매장의 사건만** 대표 이미지를 VLM에 보내 설명을 돌려줌. 인증은 `X-QNA-Token`(= `STOREOPS_QNA_TOKEN`), 한 번에 최대 5건, 다른 매장 사건은 "찾을 수 없음" |
| `app/services/vlm_describe.py` (신규) | Qwen(OpenAI 호환) 호출. 이미지를 `QWEN_VLM_IMAGE_MAX_PIXELS`(기본 147456) 이하로 줄여 전송. 장면에 보이는 것만 쓰고 신원·추측 금지, 분류 모델 판정을 사실로 단정하지 않도록 지시 |
| `app/main.py`, `requirements.txt`(`pillow`), `tests/test_internal_qna.py` | 라우터 등록, 이미지 축소, 테스트 |

**질문 Agent (`storeops_qna/`)**

| 파일 | 내용 |
| --- | --- |
| `tools.py` | 도구 `describe_events` 추가(입력 검사: 최대 5건, `E###` 형식). `query_records` 결과 문장에 사건 종류·시각·카메라 추가. 해석 서버가 없으면 사유를 그대로 알리고 내용을 지어내지 않음 |
| `agent.py` | 시스템 프롬프트에 도구 설명, `describe_events`도 앞선 기록 조회로 확인된 사건 번호만 허용, 출처·안내문 정리 |
| `config.py` | `STOREOPS_BACKEND_URL` 환경변수(질문 Agent가 백엔드의 해석 API를 부를 주소) |
| `tests/test_describe_events.py` (신규) | 가짜 LLM으로 도구 흐름 검증 |

**compose**: `docker-compose.yml`에서 백엔드에 `QWEN_VLM_BASE_URL`·`QWEN_VLM_MODEL`을 주고, 탐지 서비스가 VLM 헬스체크를 기다리던 의존성을 뺐습니다(사건 시점에 VLM을 쓰지 않으므로). `docker-compose.novlm.yml`은 VLM 서버를 아예 띄우지 않을 때의 덮어쓰기 파일입니다.

## 컴퓨터를 나눠서 쓸 때 (카메라 / 백엔드 / LLM)

탐지 서비스는 클립과 대표 이미지를 파일로 저장하고 백엔드에는 **경로만** 보내므로, 둘이 다른 컴퓨터면 같은 폴더를 보게 해야 합니다.
백엔드 컴퓨터에 공유 폴더를 만들고 카메라 컴퓨터가 그 폴더에 저장하는 방식입니다.

| 컴퓨터 | 설정 |
| --- | --- |
| **백엔드** | 공유 폴더를 만들고 compose 실행 때 `STOREOPS_OUTPUT_HOST_DIR=<그 폴더>`로 지정. `STOREOPS_AI_INGEST_TOKENS_JSON`(토큰→매장), `STOREOPS_QNA_URL`(LLM 컴퓨터 주소), VLM 서버가 다른 컴퓨터면 `STOREOPS_VLM_URL`. 탐지 서비스를 이 컴퓨터에서 돌리지 않으면 `docker compose up -d postgres backend frontend demand`처럼 `storeops-ai`를 빼고 실행 |
| **카메라(탐지 서비스)** | `ALERT_WEBHOOK_URL=http://<백엔드>:8000/api/internal/events`, `VLM_RESULT_WEBHOOK_URL=http://<백엔드>:8000/api/internal/events/{event_id}/vlm`, `EVENT_INGEST_TOKEN`(백엔드에 등록한 토큰), **`STOREOPS_OUTPUT_DIR=\\<백엔드>\<공유 폴더>`**(클립·대표 이미지가 여기에 저장됨). 사건 JSON(내부 점수 포함)은 이 컴퓨터의 `output/events`에 남음 |
| **LLM(질문 Agent)** | `STOREOPS_QNA_TOKEN`(백엔드와 같은 값), `STOREOPS_BACKEND_URL=http://<백엔드>:8000`, `STOREOPS_QNA_LLM_URL` |

- 사건 시각은 **시간대를 포함해** 보냅니다. 백엔드는 시간대 없는 시각을 UTC로 간주하므로, 한국 시간으로 도는 카메라 컴퓨터에서 시간대 없이 보내면 사건 시각이 9시간 어긋납니다(고침).
- 백엔드 컴퓨터는 8000 포트가 카메라·LLM 컴퓨터에서 열려 있어야 하고, 공유 폴더는 카메라 컴퓨터 계정에 쓰기 권한이 있어야 합니다.
- 확인 범위: 같은 컴퓨터에서 폴더 두 개를 같은 공유 폴더로 가정해(탐지 서비스는 `STOREOPS_OUTPUT_DIR`, 백엔드도 같은 폴더) 클립·이미지 저장·서빙·질문 시점 해석까지 확인했습니다. 실제 네트워크 공유(SMB)와 방화벽은 확인하지 못했습니다.

### 설정은 모두 `.env`에

비밀(토큰·비밀번호)과 컴퓨터마다 다른 주소·경로는 코드에 넣지 않고 각 컴퓨터의 `.env`에 둡니다. `.env`는 `.gitignore`에 들어 있어 저장소에 올라가지 않고, 저장소에는 값이 빈 `.env.example`만 있습니다.
- **카메라 컴퓨터**: `storeops_ai/.env`(템플릿: `storeops_ai/.env.example`). `python pipelines/path1_webcam.py`로 직접 실행하면 `config`가 이 파일을 읽습니다. 셸에 이미 설정된 환경변수가 있으면 그 값이 우선하고, 값이 빈 줄은 무시되어 기본값이 쓰입니다. 도커로 실행하면 compose가 같은 파일을 읽습니다.
- **백엔드 컴퓨터**: 루트 `.env`(`STOREOPS_AI_INGEST_TOKENS_JSON`, `STOREOPS_QNA_TOKEN`, `STOREOPS_QNA_URL`, `STOREOPS_DEMAND_TOKENS_JSON`, `POSTGRES_PASSWORD`, `STOREOPS_OUTPUT_HOST_DIR`, `STOREOPS_VLM_URL` 등).
- **LLM 컴퓨터**: `STOREOPS_QNA_TOKEN`(백엔드와 같은 값), `STOREOPS_BACKEND_URL`, `STOREOPS_QNA_LLM_URL`.
- 토큰은 랜덤 문자열을 새로 만들어 쓰고(`python -c "import secrets;print(secrets.token_urlsafe(32))"`), 공유 폴더 로그인 정보는 `.env`가 아니라 윈도우 자격 증명 관리자에 저장합니다.
- 기존 저장소의 `docker-compose.yml`, `backend/.env.example`에는 개발용 DB 비밀번호가 적혀 있으니 실제로 쓸 때는 `.env`의 값으로 바꾸세요.

## 가중치 (저장소에 포함하지 않음)

`storeops_ai/training/runs/` 아래에 같은 이름으로 둡니다(도커 빌드 컨텍스트에 포함되도록 `COPY . .` 전에 배치).

| 경로 | 설명 |
| --- | --- |
| `ours_a_final/best.pt`, `label_map.json` | A(화면 전체) S3D, 4클래스 |
| `ours_c1_final/best.pt`, `label_map.json` | C1(사람 크롭) S3D, 4클래스 |
| `ours_stack/fusion.json` | 동적 결합과 카테고리별 임계값(날짜 5-fold OOF, 재현율 목표 0.9, 전도 0.95) |

없으면 A 단독으로 동작하고, 가중치가 전혀 없으면 `I3DNotConfiguredError`를 냅니다(임의 점수를 만들지 않음).

## 환경변수

| 변수 | 위치 | 설명 |
| --- | --- | --- |
| `VLM_ON_EVENT` | 탐지 서비스 | 1이면 사건 시점에 VLM이 이미지를 읽음(기본 0) |
| `CONFIRMATIONS_REQUIRED` | 탐지 서비스(실시간 웹캠) | 연속 확인 횟수(기본 1, 이전 3) |
| `.env` 읽기 | 탐지 서비스 | `config/config.py`의 `load_env_file`이 `storeops_ai/.env`를 읽음(표준 라이브러리, 기존 환경변수 우선) |
| `STOREOPS_OUTPUT_DIR` | 탐지 서비스·백엔드 | 클립·대표 이미지 저장·읽기 폴더(공유 폴더 가능) |
| `STOREOPS_BACKEND_URL` | 질문 Agent | 백엔드 주소. 없으면 사진 해석 없이 사건 정보만 안내 |
| `QWEN_VLM_BASE_URL`, `QWEN_VLM_MODEL` | 백엔드 | 질문 시점 VLM 서버(compose에서 설정) |

## 실행

```powershell
docker compose up -d --build                                              # VLM 포함(질문 시점 사진 해석 가능)
docker compose -f docker-compose.yml -f docker-compose.novlm.yml up -d --build   # VLM 서버 없이
```

## 동작 확인 결과와 한계

- 로컬에서 실제 탐지 서비스(우리 모델)와 백엔드(메모리 저장소), 질문 서비스를 띄워 끝까지 확인했습니다: 사건 3건 저장, 점수·임계값 비어 있음, VLM "해당 없음", 클립·이미지 3장 제공, 질문 한 번에 `query_records → describe_events → 백엔드 → VLM` 순서로 호출되고 이미지는 512×288로 줄어 전달됨.
- **LLM과 VLM은 실제 서버가 아니라 OpenAI 호환 가짜 서버**로 확인했습니다. 실제 Qwen 서버의 응답 품질, 질문 Agent 실제 LLM의 도구 선택은 확인하지 못했습니다.
- 도커 이미지 빌드와 컨테이너 기동, 프런트 화면 확인은 하지 못했습니다. 도커 이미지는 CPU용 torch라서 CPU 추론 시간(이 PC 기준 모델 2개 합쳐 2초 창당 약 0.6초)을 배포 환경에서 확인해야 합니다.
- 평가 한계: 공식 validation을 여러 번 보았으므로 블라인드가 아니며, 새 환경(강의실) 성능은 별도 측정이 필요합니다. 오경보는 개발 평가 기준 클래스당 시간당 약 2건이므로 경보 문구는 "의심"으로 둡니다.
- 프런트의 점수 영역은 점수가 비면 자동으로 숨겨집니다. 다만 같은 화면에 있는 "현재 저장소의 학습 가중치에는 싸움 클래스가 없어 점수는 0입니다." 안내 문장은 점수 영역 안에 있어 함께 사라집니다(프런트는 수정하지 않음).
