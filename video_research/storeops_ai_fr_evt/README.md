# StoreOps AI — FR-EVT-01~16

매장 CCTV 영상을 분석해 이상행동(폭행/파손/절도 등)을 탐지하고, 카메라 연결 끊김을 감시하는 이벤트 탐지 백엔드입니다. FastAPI 기반이며 크게 두 개의 독립된 파이프라인(Path 1, Path 2)으로 구성됩니다.

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [동작 개요](#동작-개요-path-1--path-2)
- [적용 범위 (요구사항 매핑)](#적용-범위)
- [중요: I3D 관련 제약사항](#중요-i3d)
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
│   ├── yolo_detector.py       # YOLO + BoT-SORT 래퍼 (FR-EVT-01~02)
│   ├── i3d_classifier.py      # I3D 행동분류 어댑터 — 실제 가중치 미연결 상태 (아래 참고)
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
└── output/                     # 실행 결과물 (아래 "출력" 항목 참고)
```

## 동작 개요 (Path 1 / Path 2)

두 파이프라인은 서로 독립적으로 동작하며, 하나가 실패해도 다른 하나에 영향을 주지 않습니다.

- **Path 1 (행동 분류)**: 영상 파일을 입력받아 `YOLO 탐지 → BoT-SORT 추적 → 사람별 crop → 2~4초 클립 묶음 → I3D 행동분류(7개 클래스) → 0.60 이상이면 사건후보 생성 → 1차 알림 → VLM 2차 검증` 순서로 처리합니다. CLI로 영상 파일 하나를 직접 분석할 때 씁니다.
- **Path 2 (카메라 끊김 감시)**: 영상 내용은 전혀 보지 않고, 카메라별로 마지막 프레임 수신 시각만 추적합니다. FastAPI 엔드포인트로 프레임 수신을 알리고(`/path2/frame`), 주기적으로 끊김 여부를 체크(`/path2/check`)하는 방식으로, 실제 카메라 연동 시스템에서 주기적으로 호출해주는 구조입니다.

## 적용 범위

### Path 1 — 행동 분류
- FR-EVT-01 YOLO 사람 위치/탐지점수
- FR-EVT-02 BoT-SORT 추적번호
- FR-EVT-03 동일 ID crop, 폭행/파손 주변 확대
- FR-EVT-04 2~4초 묶음
- FR-EVT-05 I3D 7개 클래스 점수
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

## 중요: I3D

실제 I3D 학습 가중치 파일과 입력 형식은 제공되지 않았기 때문에
`models/i3d_classifier.py`는 **실제 점수를 만들어내지 않는 어댑터**로 두었습니다.

즉, I3D 가중치가 없는 상태에서 임의의 점수를 만들어 사건을 발생시키지 않습니다.
실제 checkpoint가 준비되면 `I3DClassifier.predict()`를 해당 모델의 입력 형식에 맞게 연결하면 FR-EVT-05/06이 실제 동작합니다.

## 설치

**Python 3.9 이상** 권장 (테스트 환경: 3.9). 가상환경 사용을 권장합니다.

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

`requirements.txt`에는 `fastapi`, `uvicorn`, `ultralytics`(YOLO), `opencv-python`, `numpy`, `pydantic`이 포함되어 있습니다. `ultralytics`가 `torch`를 함께 설치하므로 최초 설치 시 다소 시간이 걸릴 수 있습니다. GPU를 쓰려면 `pip install -r requirements.txt` 이후 CUDA 버전에 맞는 `torch`를 별도로 재설치해야 합니다 (기본은 CPU 버전이 설치됩니다).

## 환경 변수 / 설정값 (`config/config.py`)

아래 값들은 모두 환경변수로 덮어쓸 수 있고, 지정하지 않으면 기본값이 사용됩니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `YOLO_MODEL_PATH` | `yolo11n.pt` | YOLO 가중치 경로 (FR-EVT-01) |
| `TRACKER_CONFIG` | `botsort.yaml` | BoT-SORT 추적 설정 (FR-EVT-02) |
| `CONF_THRESHOLD` | `0.25` | 사람 탐지 최소 신뢰도 |
| `CLIP_MIN_SEC` / `CLIP_MAX_SEC` | `2.0` / `4.0` | 후보 클립 최소/최대 길이 (FR-EVT-04) |
| `EXPAND_X` / `EXPAND_Y` | `0.25` / `0.25` | 폭행/파손 등 확대 crop 시 여백 비율 |
| `EVENT_THRESHOLD` | `0.60` | 사건후보 생성 최소 점수 (FR-EVT-06) |
| `EVENT_VIDEO_SEC` | `10.0` | 사건 영상 길이(초) (FR-EVT-08) |
| `CAMERA_DISCONNECT_SEC` | `60.0` | 카메라 끊김 판정 기준 시간(초) (FR-EVT-10) |
| `VLM_MAX_WORKERS` | `2` | VLM 비동기 검증 동시 실행 개수 |

예시 (Path1 실행 시 신뢰도 임계값을 바꾸고 싶을 때):
```bash
# Windows PowerShell
$env:CONF_THRESHOLD="0.4"; python -m pipelines.path1_behavior --video sample.mp4

# Mac/Linux
CONF_THRESHOLD=0.4 python -m pipelines.path1_behavior --video sample.mp4
```

## AI Hub 유기행동 스모크 테스트

`../유기`의 XML에서 학습 47건·검증 16건의 유기 구간 manifest를 만들고,
지정한 영상에 BoT-SORT 사람 ID와 XML의 유기 행동 구간을 겹쳐 확인합니다.

```powershell
python -m pipelines.aihub_abandonment --smoke-video "../유기/AI_validation/AI_validation/videos/영상파일.mp4"
```

결과는 `output/aihub_abandonment/`에 저장됩니다. 유기는 별도 물건 상태기계 없이
공통 7분류 I3D의 한 클래스로 학습합니다.

## Path 1 테스트

CLI로 영상 파일 하나를 직접 분석합니다. 프로젝트 루트에서 실행하세요.

**인자**

| 인자 | 필수 | 설명 |
|---|---|---|
| `--video` | ✅ | 분석할 영상 파일 경로 |
| `--camera-id` | ❌ (기본 `CAM001`) | 카메라 식별자 |
| `--i3d-weight` | ❌ | I3D 가중치 경로 (없으면 행동분류는 비활성 상태로 동작) |
| `--debug` | ❌ | 디버그 로그 출력 |

**실행 예시**
```bash
python -m pipelines.path1_behavior --video "sample.mp4" --camera-id CAM001 --debug
```

결과는 표준출력에 JSON 배열(`EventCandidate` 리스트)로 출력되고, 동시에 `output/` 아래에 클립/이미지/이벤트 JSON이 저장됩니다.

> I3D가 연결되지 않은 상태(`--i3d-weight` 미지정)에서는 YOLO 탐지 → 추적 → 후보 crop/클립 생성까지는 정상적으로 확인할 수 있지만, `I3DNotConfiguredError`가 발생해 **실제 행동 사건(FR-EVT-06 이후)은 생성되지 않습니다.** 자세한 내용은 아래 "중요: I3D" 항목을 참고하세요.

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
│   ├── candidate_*.mp4          # 사건 후보 원본 클립 (2~4초)
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
output
```

`yolo11n.pt`는 기존 파일을 프로젝트 루트에 유지하세요. (이 ZIP에도 동일 파일이 포함되어 있습니다.)

압축을 푼 뒤 위의 "설치" 항목대로 가상환경을 새로 만들고 `pip install -r requirements.txt`를 실행하면 됩니다. (이 ZIP에는 원래 있던 `venv` 폴더는 포함되어 있지 않습니다 — 다른 PC의 경로가 박혀 있어 그대로 쓸 수 없기 때문입니다.)

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| Path1 실행 시 `I3DNotConfiguredError` | 정상입니다. 실제 I3D 가중치가 없는 상태이므로 의도적으로 발생시킵니다. `--i3d-weight`로 실제 checkpoint를 연결하면 해결됩니다. |
| `pip install` 이 매우 오래 걸림 / 용량이 큼 | `ultralytics` 설치 시 `torch`가 함께 설치되기 때문입니다 (정상). 네트워크 상황에 따라 수 분 걸릴 수 있습니다. |
| YOLO 실행 시 `botsort.yaml` 관련 오류 | `ultralytics` 패키지 내장 기본 설정을 사용하므로 별도 파일 생성 없이 그대로 실행하면 됩니다. 커스텀하려면 동일한 이름의 yaml을 프로젝트에 두고 경로를 맞추세요. |
| `output/` 폴더에 아무 것도 생성되지 않음 | Path1은 사건후보가 하나도 없으면(I3D 미연결 등) 이벤트 JSON을 생성하지 않습니다. 클립/이미지는 후보 단계에서부터 생성되므로, 그것도 없다면 YOLO가 사람을 하나도 탐지하지 못한 경우입니다 (`CONF_THRESHOLD`를 낮춰보세요). |
| Path2 끊김 사건이 계속 재생성됨 | `/path2/frame`을 호출해 정상 수신 상태로 되돌리기 전까지는 `event_created_for_disconnect` 플래그로 중복 생성을 막습니다. 계속 생성된다면 매 호출마다 새로운 `CameraConnectionMonitor` 인스턴스가 만들어지고 있는 건 아닌지(서버 재시작 등) 확인하세요. 현재 구현은 인메모리 상태이므로 서버를 재시작하면 상태가 초기화됩니다. |
