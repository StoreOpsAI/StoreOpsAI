# StoreOps AI — FR-EVT-01~16

매장 CCTV 영상을 분석해 이상행동(폭행/파손/절도 등)을 탐지하고, 카메라 연결 끊김을 감시하는 이벤트 탐지 백엔드입니다. FastAPI 기반이며 크게 두 개의 독립된 파이프라인(Path 1, Path 2)으로 구성됩니다.

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [동작 개요](#동작-개요-path-1--path-2)
- [적용 범위 (요구사항 매핑)](#적용-범위)
- [행동분류 모델 연결 (FR-EVT-05)](#행동분류-모델-연결-fr-evt-05)
- [설치](#설치)
- [환경 변수 / 설정값](#환경-변수--설정값-configconfigpy)
- [Path 1 실행 (CLI)](#path-1-테스트)
- [Path 2 실행 (API 서버)](#path-2-테스트)
- [출력 구조](#출력)
- [Windows 적용](#windows-적용)
- [트러블슈팅](#트러블슈팅)

## 프로젝트 구조

```
storeops_ai_fr_evt/
├── main.py                    # FastAPI 앱 엔트리포인트 (Path1/Path2 라우터 등록)
├── requirements.txt
├── yolo11n.pt                 # YOLOv11 사람 탐지 사전학습 가중치
│
├── api/
│   ├── routes_path1.py        # Path1(행동 분류) 관련 엔드포인트
│   └── routes_path2.py        # Path2(카메라 끊김) 관련 엔드포인트
│
├── config/
│   └── config.py               # 임계값, 경로 등 전역 설정 (모두 환경변수로 override 가능)
│
├── models/
│   ├── yolo_detector.py       # YOLO + ByteTrack 래퍼 (FR-EVT-01~02)
│   ├── i3d_classifier.py      # 학습된 S3D 체크포인트 로드 + 추론 (FR-EVT-05)
│   └── vlm_analyzer.py        # VLM(비전-언어 모델) 2차 검증 연동부
│
├── pipelines/
│   ├── path1_behavior.py      # 행동 분류 파이프라인 (CLI 실행형)
│   └── path2_connection.py    # 카메라 연결 상태 모니터 (API 호출형)
│
├── schemas/
│   └── event_schema.py        # 사건 후보(EventCandidate) pydantic 모델
│
├── utils/
│   ├── clip_manager.py        # 프레임 저장 / 대표 이미지 추출
│   ├── event_manager.py       # 사건 ID 채번(E001, E002 ...) 및 JSON 저장
│   └── notification.py        # 1차 알림 발송
│
├── tools/
│   └── verify_i3d.py          # 학습된 모델만 단독 확인 (YOLO/추적 없이)
│
├── training/                   # 학습 산출물 (handoff_0921)
│   ├── manifests/             # train/val 클립 목록 (학습에 쓴 데이터 목록. 프레임 이미지 자체는 포함되지 않음)
│   ├── runs/
│   │   ├── mc_stage1/         # 1단계: 백본 동결, head만 학습 (val_acc 0.615)
│   │   └── mc_stage2/         # 2단계: 전체 미세조정 (val_acc 0.750) <- 기본 사용
│   └── scripts/               # extract_multiclass.py(프레임 추출), train_i3d.py(학습), check_shortcut_risk.py, run_pipeline.sh
│
└── output/                     # 실행 결과물 (아래 "출력" 항목 참고)
```

## 동작 개요 (Path 1 / Path 2)

두 파이프라인은 서로 독립적으로 동작하며, 하나가 실패해도 다른 하나에 영향을 주지 않습니다.

- **Path 1 (행동 분류)**: 파일 분석과 실시간 웹캠 모두 `YOLO 탐지 → ByteTrack 추적 → 화면 전체 S3D 행동분류 → 사건후보 → 1차 알림 → VLM 2차 검증` 순서로 처리합니다. 사건 영상은 판정 시점 전 5초/후 5초로 확정하며, VLM 결과·실패 상태는 사건 JSON에 비동기로 반영됩니다.
- **Path 2 (카메라 끊김 감시)**: 영상 내용은 전혀 보지 않고, 카메라별 마지막 프레임 수신 시각만 추적합니다. `/path2/streams`에 RTSP/HTTP 주소를 등록하면 수신 워커와 내부 스케줄러가 자동으로 끊김을 감시합니다. 외부 수신기는 기존 `/path2/frame` API를 계속 사용할 수 있습니다.

## 적용 범위

### Path 1 — 행동 분류
- FR-EVT-01 YOLO 사람 위치/탐지점수
- FR-EVT-02 ByteTrack 추적번호
- FR-EVT-03 화면 전체 입력(사람 track ID는 사건 연계용), 파손/폭행은 전체 장면 증거 보존
- FR-EVT-04 4초 묶음, 2초 overlap
- FR-EVT-05 S3D 6개 학습 클래스 점수 + 폭행 예약 클래스
- FR-EVT-06 0.60 이상 사건 후보 + E001 형식 번호
- FR-EVT-07 VLM과 독립된 1차 알림
- FR-EVT-08 10초 사건 영상 + 카메라 번호
- FR-EVT-13 처음/가운데/끝 이미지 3장
- FR-EVT-14 VLM 3항목 검사
- FR-EVT-15 VLM 지연/실패가 사건/알림에 영향 없음
- FR-EVT-16 끊김 사건은 VLM 미실행

### Path 2 — 카메라 끊김
- FR-EVT-09 연결상태/마지막 프레임 시각만 확인
- FR-EVT-10 기준시간 초과 시 1회 생성
- FR-EVT-11 중복 방지 + 재연결 시 상태 종료
- FR-EVT-12 끊김 시간/기준시간 기록, 영상/점수 비움

## 행동분류 모델 연결 (FR-EVT-05)

`training/runs/mc_stage2/best.pt` (torchvision **S3D**, Kinetics-400 사전학습 후 미세조정)가 기본으로 연결되어 있습니다.
`--i3d-weight` 또는 `I3D_WEIGHT_PATH` 로 다른 체크포인트(예: `training/runs/mc_stage1/best.pt`)를 지정할 수 있습니다.

| 항목 | 내용 |
|---|---|
| 학습 클래스 | abandon(유기) / broken(파손) / fall(전도) / fire(방화) / theft(절도) / normal(정상) — **6종** |
| **폭행** | 학습하지 않았습니다. 이 모델에서는 폭행 점수가 항상 0.0이라 **폭행 사건은 발급되지 않습니다.** |
| 입력 | 24프레임 × 224×224 (프레임을 정사각형으로 리사이즈, Kinetics 정규화) — 학습과 동일 |
| 학습 검증 정확도 | stage2 0.750 (52클립). 클립 수가 적어 변동이 큽니다 |

**입력 방식**

운영 모드는 `ACTION_INPUT_MODE=full`로 고정했습니다. 화면 전체를 3fps로 버퍼링해 **4초 창을 2초 간격(50% overlap)** 으로 S3D에 전달합니다. 사람이 한 명도 추적되지 않은 창은 분류하지 않으며, 사건의 `track_ids`에는 해당 창에서 나타난 ID를 기록합니다. 사람 crop은 증거용으로만 남기고 분류 입력에는 사용하지 않습니다.

같은 카테고리 사건은 `EVENT_COOLDOWN_SEC`(30초) 동안 다시 발급하지 않습니다.
영상이 `ACTION_WINDOW_SEC`보다 짧으면 영상 전체를 1회 분류합니다.

**모델만 단독으로 확인하기**

```bash
python -m tools.verify_i3d                                   # 체크포인트 정보
python -m tools.verify_i3d --video sample.mp4                # 영상 전체 1회 분류
python -m tools.verify_i3d --video sample.mp4 --window-sec 30 --stride-sec 5   # 구간별 분류
python -m tools.verify_i3d --frames-dir some_clip_folder     # extract_multiclass.py 가 만든 클립 폴더
```

**알아둘 점 (학습 데이터 한계)**

- 학습 데이터가 AI Hub CCTV 3fps 영상이고, 학습 클립의 실제 시간 폭은 정상·절도 8초, 나머지 클래스 20~50초로 클래스마다 달랐습니다.
  프레임 간격이 클래스 단서로 학습됐을 가능성이 있어, 실제 카메라 영상에서는 `ACTION_WINDOW_SEC` 조정이 필요할 수 있습니다.
- `EVENT_THRESHOLD`(0.60)는 학습 데이터가 적은 6-way softmax 기준으로는 낮을 수 있습니다. 정상 영상에서 오탐이 나면 카테고리별로 올려 보세요 (`config/config.py` 의 `CATEGORY_THRESHOLDS`).
- 재학습: `training/scripts/run_pipeline.sh` (원본 영상/라벨 XML 필요. 이 패키지에는 포함되어 있지 않습니다).

## 설치

**Python 3.10 이상** 필요 (torch 요구사항). 가상환경 사용을 권장합니다.

```bash
# 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

`requirements.txt`에는 `fastapi`, `uvicorn`, `ultralytics`(YOLO), `opencv-python`, `numpy`, `pydantic`, `torch`, `torchvision`이 포함되어 있습니다. 최초 설치 시 다소 시간이 걸릴 수 있습니다. GPU를 쓰려면 `pip install -r requirements.txt` 이후 CUDA 버전에 맞는 `torch`를 별도로 재설치해야 합니다 (기본은 CPU 버전이 설치됩니다).

## 환경 변수 / 설정값 (`config/config.py`)

아래 값들은 모두 환경변수로 덮어쓸 수 있고, 지정하지 않으면 기본값이 사용됩니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `YOLO_MODEL_PATH` | `yolo11n.pt` | YOLO 가중치 경로 (FR-EVT-01) |
| `TRACKER_CONFIG` | `bytetrack.yaml` | ByteTrack 추적 설정 (FR-EVT-02) |
| `CONF_THRESHOLD` | `0.25` | 사람 탐지 최소 신뢰도 |
| `CLIP_MIN_SEC` / `CLIP_MAX_SEC` | `2.0` / `4.0` | 후보 클립 최소/최대 길이 (FR-EVT-04) |
| `EXPAND_X` / `EXPAND_Y` | `0.50` / `0.50` | 폭행/파손 증거 crop 시 bbox 사방 여백 비율 |
| `FALL_EVENT_THRESHOLD` | `0.80` | 전도 사건 기준 |
| `BROKEN_EVENT_THRESHOLD` / `FIRE_EVENT_THRESHOLD` | `0.75` / `0.85` | 파손/방화 사건 기준 |
| `ABANDON_EVENT_THRESHOLD` / `THEFT_EVENT_THRESHOLD` | `0.75` / `0.80` | 유기/절도 사건 기준 |
| `NORMAL_ANOMALY_MARGIN` | `0.20` | 비정상 최고점이 정상보다 높아야 하는 최소 차이 |
| `EVENT_VIDEO_SEC` | `10.0` | 사건 영상 길이(초) (FR-EVT-08) |
| `EVENT_VIDEO_PRE_SEC` / `EVENT_VIDEO_POST_SEC` | `5.0` / `5.0` | 판정 시점 전/후 영상 길이. 후행 프레임 수신 뒤 확정 |
| `CAMERA_DISCONNECT_SEC` | `60.0` | 카메라 끊김 판정 기준 시간(초) (FR-EVT-10) |
| `VLM_MAX_WORKERS` | `2` | VLM 비동기 검증 동시 실행 개수 |
| `OPENAI_VLM_MODEL` | `gpt-5` | `OPENAI_API_KEY` 설정 시 사용할 VLM 모델 |
| `ALERT_WEBHOOK_URL` | 비어 있음 | 사건 JSON을 POST할 1차 알림 웹훅 |
| `I3D_WEIGHT_PATH` | `training/runs/mc_stage2/best.pt` | 행동분류 체크포인트 (FR-EVT-05) |
| `I3D_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `ACTION_WINDOW_SEC` / `ACTION_WINDOW_OVERLAP_SEC` | `4.0` / `2.0` | 화면 전체 분류 창 및 겹침 길이 |
| `I3D_BUFFER_FPS` | `3.0` | 분류용 프레임 버퍼 속도 (학습 영상이 3fps) |
| `EVENT_COOLDOWN_SEC` | `30.0` | 같은 카테고리 사건 재발급 금지 시간(초) |

예시 (Path1 실행 시 신뢰도 임계값을 바꾸고 싶을 때):
```bash
# Windows PowerShell
$env:CONF_THRESHOLD="0.4"; python -m pipelines.path1_behavior --video sample.mp4

# Mac/Linux
CONF_THRESHOLD=0.4 python -m pipelines.path1_behavior --video sample.mp4
```

## Path 1 테스트

CLI로 영상 파일 하나를 직접 분석합니다. 프로젝트 루트에서 실행하세요.

**인자**

| 인자 | 필수 | 설명 |
|---|---|---|
| `--video` | ✅ | 분석할 영상 파일 경로 |
| `--camera-id` | ❌ (기본 `CAM001`) | 카메라 식별자 |
| `--i3d-weight` | ❌ | I3D 가중치 경로 (생략하면 `training/runs/mc_stage2/best.pt`) |
| `--debug` | ❌ | 디버그 로그 출력 |

**실행 예시**
```bash
python -m pipelines.path1_behavior --video "sample.mp4" --camera-id CAM001 --debug
```

결과는 표준출력에 JSON 배열(`EventCandidate` 리스트)로 출력되고, 동시에 `output/` 아래에 클립/이미지/이벤트 JSON이 저장됩니다.

> 분류 로그(`[I3D] t=..s 카테고리 confidence | 점수`)가 분류 1회마다 출력됩니다. 임계값 미만이거나 '정상'이면 사건이 발급되지 않습니다.
> 가중치 파일이 없으면(`I3D_WEIGHT_PATH` 잘못 지정 등) 탐지/추적까지만 수행하고 사건은 발급하지 않습니다.

API로도 호출할 수 있습니다: `POST /path1/analyze` (`{"video_path": "sample.mp4", "camera_id": "CAM001"}`).

## Path 2 테스트

FastAPI 서버를 띄워서 API 호출로 테스트합니다.

**서버 실행**
```bash
uvicorn main:app --reload
```
기본적으로 `http://127.0.0.1:8000` 에서 서비스되며, `http://127.0.0.1:8000/docs` 에서 Swagger UI로 바로 테스트해볼 수 있습니다.

**엔드포인트**

| Method | Path | 설명 |
|---|---|---|
| GET | `/` | 서비스 상태 확인 |
| POST | `/path2/frame` | 해당 카메라가 프레임을 정상 수신했음을 알림 (마지막 수신 시각 갱신) |
| POST | `/path2/check` | 마지막 수신 이후 `CAMERA_DISCONNECT_SEC`(기본 60초)를 초과했는지 검사, 초과 시 끊김 사건 1건 생성 |
| GET | `/path2/state/{camera_id}` | 해당 카메라의 현재 연결 상태 조회 |
| POST | `/path2/streams` | `camera_id`, `stream_url`로 RTSP/HTTP 수신 워커 등록 |
| DELETE | `/path2/streams/{camera_id}` | 수신 워커 중지 |
| WebSocket | `/path1/ws/events` | 새 사건과 후속 변경(VLM/영상/점주 상태)을 React에 전달 |
| PATCH | `/path1/events/{event_id}/status` | 점주가 `확인` 또는 `오탐`으로 사건 상태 변경 |

**React 실시간 메시지 형식**

WebSocket 연결 주소는 `ws://서버주소:8000/path1/ws/events`이다. 서버가 보내는 메시지는 아래 두 종류이며, `event`에는 사건 JSON 전체가 들어간다.

```json
{"type":"event.created","occurred_at":"2026-09-22T...Z","event":{"event_id":"E015","status":"사건후보"}}
{"type":"event.updated","occurred_at":"2026-09-22T...Z","event":{"event_id":"E015","status":"확인"}}
```

점주 상태 변경 요청 예시는 다음과 같다.

```json
PATCH /path1/events/E015/status
{"status":"확인"}
```

`CORS_ALLOW_ORIGINS` 환경변수에 React 개발/운영 URL을 쉼표로 구분해 지정한다. 현재 WebSocket/API에는 사용자 인증이 없으므로 외부 운영 배포 전 인증을 추가해야 한다.

**요청 예시**

프레임 수신 알림:
```bash
curl -X POST http://127.0.0.1:8000/path2/frame \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "CAM001"}'
```

끊김 여부 확인:
```bash
curl -X POST http://127.0.0.1:8000/path2/check \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "CAM001"}'
```

상태 조회:
```bash
curl http://127.0.0.1:8000/path2/state/CAM001
```

`timestamp`(unix epoch, 초) 필드를 요청 본문에 함께 보내면 실제 시간 대신 해당 시각 기준으로 테스트할 수 있습니다 (자동화 테스트 시 유용).

## 출력

```
output/
├── clips/
│   ├── E001_2to4sec.mp4         # 사건별 짧은 클립
│   └── E001_10sec.mp4           # 사건 영상 (FR-EVT-08, 10초)
├── representative_images/
│   ├── E001_1.jpg               # 사건 시작 시점 대표 이미지
│   ├── E001_2.jpg               # 사건 중간 시점
│   └── E001_3.jpg               # 사건 끝 시점 (VLM 3항목 검사용, FR-EVT-13~14)
└── events/
    └── E001.json                # EventCandidate 전체 필드 (schemas/event_schema.py 참고)
```

사건 ID는 `output/events/` 안의 기존 `E***.json` 파일 중 가장 큰 번호 다음 번호로 자동 채번됩니다 (`E999` 다음은 `E1000`). 카메라 끊김 사건은 `event_video_path`, `representative_images`, `scores`가 비워진 채로 `disconnect_seconds` / `disconnect_threshold_seconds`만 채워져 저장됩니다.

## Windows 적용

기존 프로젝트를 삭제하지 말고 백업한 뒤 이 ZIP의 내용을 `C:\storeops_ai`에 복사하세요.

필요한 폴더:
```
config
models
pipelines
schemas
utils
api
tools
training      <- 학습된 모델(runs/)이 여기 있습니다. 빠지면 행동분류가 비활성화됩니다.
output
```

`yolo11n.pt`는 기존 파일을 프로젝트 루트에 유지하세요. (이 ZIP에도 동일 파일이 포함되어 있습니다.)

압축을 푼 뒤 위의 "설치" 항목대로 가상환경을 새로 만들고 `pip install -r requirements.txt`를 실행하면 됩니다. (이 ZIP에는 원래 있던 `venv` 폴더는 포함되어 있지 않습니다 — 다른 PC의 경로가 박혀 있어 그대로 쓸 수 없기 때문입니다.)

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `I3D 가중치 없음 ... 행동분류 비활성 상태` 경고 | `training/runs/mc_stage2/best.pt` 가 없거나 `I3D_WEIGHT_PATH` 가 틀렸습니다. |
| 영상이 길면 메모리 부족 | Path1은 분석 중 모든 원본/주석 프레임을 메모리에 보관합니다 (기존 구조). 긴 영상은 잘라서 분석하세요. |
| `pip install` 이 매우 오래 걸림 / 용량이 큼 | `ultralytics` 설치 시 `torch`가 함께 설치되기 때문입니다 (정상). 네트워크 상황에 따라 수 분 걸릴 수 있습니다. |
| YOLO 실행 시 `bytetrack.yaml` 관련 오류 | `ultralytics` 패키지 내장 기본 설정을 사용하므로 별도 파일 생성 없이 그대로 실행하면 됩니다. 커스텀하려면 동일한 이름의 yaml을 프로젝트에 두고 경로를 맞추세요. |
| `output/` 폴더에 아무 것도 생성되지 않음 | Path1은 사건이 하나도 발급되지 않으면 이벤트 JSON/클립/이미지를 만들지 않습니다. 분류 로그의 카테고리·점수를 확인하세요 (정상으로 분류되었거나 임계값 미만). 로그 자체가 없으면 YOLO가 사람을 탐지하지 못한 경우입니다 (`CONF_THRESHOLD`를 낮춰보세요). |
| Path2 끊김 사건이 계속 재생성됨 | `/path2/frame`을 호출해 정상 수신 상태로 되돌리기 전까지는 `event_created_for_disconnect` 플래그로 중복 생성을 막습니다. 계속 생성된다면 매 호출마다 새로운 `CameraConnectionMonitor` 인스턴스가 만들어지고 있는 건 아닌지(서버 재시작 등) 확인하세요. 현재 구현은 인메모리 상태이므로 서버를 재시작하면 상태가 초기화됩니다. |

## FR-EVT Path 1 실시간 웹캠 테스트

학습된 `training/runs/mc_stage2/best.pt`를 실제 Path 1에 연결했다.

실시간 흐름:

`웹캠 → YOLO 사람 bbox → ByteTrack Track ID → S3D(S3D checkpoint) → 임계값 0.60 → EventCandidate(E001...)`

### 웹캠 실행

```bat
cd C:\storeops_ai
venv\Scripts\activate
python pipelines\path1_webcam.py --camera 0 --debug
```

- 기본 `full` 모드: 학습 데이터와 같은 화면 전체 입력을 사용하므로 기본 권장
이벤트가 임계값을 넘으면 `output/events/E001.json` 형태로 저장되고, `output/clips/`에 2~4초 클립/10초 사건 영상, `output/representative_images/`에 대표 이미지 3장이 생성된다.

현재 학습 체크포인트는 `abandon`, `broken`, `fall`, `fire`, `normal`, `theft` 6개 클래스이며 `fight(폭행)`은 학습되지 않았다. 따라서 폭행 점수는 0.0이다.
