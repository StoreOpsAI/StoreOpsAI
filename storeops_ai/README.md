# StoreOps AI CCTV 이벤트 탐지

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/FastAPI-Detection-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Computer%20Vision-YOLO%20%2B%20S3D-4C8BF5" alt="YOLO and S3D">
</p>

매장 영상에서 사람을 추적하고 장면 단위 행동 점수를 계산해 이벤트 후보를
저장하는 FastAPI 서비스입니다. 별도 Path 2 런타임은 RTSP/HTTP 카메라의
프레임 수신 시각만 감시합니다. 운영 카테고리(v1.0)는 정상, 쓰러짐,
쓰레기 투기, 절도이며, 학습하지 않은 싸움·파손·방화는 서비스 카테고리로
노출하지 않습니다. 행동분류는 화면 전체(A)와 사람 크롭(C1) S3D를 동적
결합해 사용합니다(자세한 내용은 [INTEGRATION_V1.0.md](INTEGRATION_V1.0.md)).

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [동작 개요](#동작-개요-path-1--path-2)
- [적용 범위 (요구사항 매핑)](#적용-범위)
- [행동분류 모델 연결 (FR-EVT-05)](#행동분류-모델-연결-fr-evt-05)
- [설치](#설치)
- [환경 변수 / 설정값](#환경-변수--설정값-configconfigpy)
- [Path 1 테스트](#path-1-테스트)
- [Path 2 테스트](#path-2-테스트)
- [출력 구조](#출력)
- [Windows 로컬 실행](#windows-로컬-실행)
- [트러블슈팅](#트러블슈팅)

## 프로젝트 구조

    storeops_ai/
    ├── main.py                    # FastAPI 앱, 런타임·자동 입력 감시기 시작
    ├── requirements.txt
    ├── yolo11n.pt                 # YOLOv11 사람 탐지 가중치
    ├── api/                        # Path 1·Path 2 API
    ├── config/                     # 임계값, 경로, 모델 설정
    ├── models/                     # YOLO, S3D 분류기, VLM 연동
    ├── pipelines/                  # 행동 분석과 카메라 연결 감시
    ├── services/                   # 스트림 런타임, 입력 감시, WebSocket
    ├── schemas/                    # 이벤트 스키마
    ├── tools/                      # S3D·VLM 확인 및 WSL vLLM 실행 도구
    ├── training/                   # 학습 산출물과 스크립트
    ├── webcam_i3d.py               # 독립 웹캠 분류 점검 도구
    └── output/                     # 실행 중 생성되는 클립·이미지·이벤트

## 동작 개요 (Path 1 / Path 2)

FastAPI 시작 시 Path 2 감시 스케줄러와 `input/` 폴더 감시기가 함께 시작됩니다.

- **Path 1 (행동 분석)**: `POST /path1/analyze` 또는 `input/` 폴더에 안정적으로
  기록된 영상으로 분석합니다. YOLO/ByteTrack으로 사람을 확인하고, 사람 ID가
  하나 이상 있는 구간에 한해 전체 장면의 4초 창을 2초 간격으로 분류합니다.
  임계값 초과 시 사건 JSON·클립·대표 이미지 저장과 1차 알림을 진행하고,
  Qwen VLM 검증 및 후속 영상 저장은 비동기로 반영합니다.
- **Path 2 (카메라 연결 감시)**: 프레임 내용은 분석하지 않고 마지막 수신 시각을
  관리합니다. `/path2/streams` 등록 시 프레임 수신 워커와 감시 스케줄러를
  사용합니다. 외부 수신기는 `/path2/frame`을 호출할 수 있습니다. 연결 상태는
  메모리 기반이며 서버 재시작 시 초기화됩니다.

## 적용 범위

### Path 1 --- 행동 분류

- FR-EVT-01 YOLO 사람 위치와 탐지 점수
- FR-EVT-02 ByteTrack 추적 번호
- FR-EVT-03 화면 전체 입력, 추적 ID는 사건 연계용
- FR-EVT-04 4초 분류 창, 2초 간격(50% 겹침), 분류 버퍼 3fps
- FR-EVT-05 정상·쓰러짐·쓰레기 투기·절도 4개 서비스 점수(v1.0, A+C1 동적 결합)
- FR-EVT-06 카테고리별 임계값 초과 시 사건 후보 발급
- FR-EVT-07 VLM과 독립된 1차 알림
- FR-EVT-08 10초 사건 영상과 카메라 번호
- FR-EVT-13 처음·가운데·끝 대표 이미지 최대 3장

### Path 2 --- 카메라 끊김

- FR-EVT-09 연결상태/마지막 프레임 시각만 확인
- FR-EVT-10 기준시간 초과 시 1회 생성
- FR-EVT-11 중복 방지 + 재연결 시 상태 종료
- FR-EVT-12 끊김 시간/기준시간 기록, 영상/점수 비움

## 행동분류 모델 연결 (FR-EVT-05)

기본 체크포인트는 torchvision S3D 두 개입니다. 화면 전체 모델
`training/runs/ours_a_final/best.pt`(A)와 사람 크롭 모델
`training/runs/ours_c1_final/best.pt`(C1)를 `training/runs/ours_stack/fusion.json`의
동적 결합 `z = (a|a| + b|b|) / (|a| + |b|)`(학습 파라미터 없음)으로 합칩니다.
임계값도 이 파일(날짜 5-fold OOF로 정한 값)에서 읽습니다. 가중치 지정은
CLI의 `--i3d-weight` 또는 `I3D_WEIGHT_PATH`를 사용하며, 기본 경로가 아닌
가중치를 지정하면 A 단독으로 동작합니다. 체크포인트 라벨을 서비스
카테고리로 다음처럼 변환합니다.

| 학습 라벨 | 서비스 점수 |
| --------- | ----------- |
| `normal`  | 정상        |
| `fall`    | 쓰러짐      |
| `abandon` | 쓰레기 투기 |
| `theft`   | 절도        |

싸움·파손·방화는 학습하지 않아 서비스 점수에서 제외합니다. 모델 입력은
24프레임을 균일 샘플링해 224×224로 리사이즈하고 Kinetics 정규화를
적용합니다.

**입력 방식**

운영 모드는 `ACTION_INPUT_MODE=full`로 고정했습니다. 화면 전체를 3fps로
버퍼링해 **4초 창을 2초 간격(50% overlap)** 으로 S3D에 전달합니다.
사람 유무와 관계없이 모든 창을 분류하며(쓰러져 추적이 끊긴 사람, 사람이
떠난 직후의 유기를 놓치지 않기 위해), 사건의 `track_ids`에는 해당 창에서
나타난 ID를 기록합니다. C1 입력은 창 안 모든 사람 박스의 합집합을 넓혀
원본 프레임에서 자른 크롭이며(`CROP_LARGEST_PERSON=1`이면 가장 큰 한 명),
사람이 없으면 화면 전체를 씁니다.

같은 카테고리 사건은 `EVENT_COOLDOWN_SEC`(30초) 동안 다시 발급하지
않습니다. 영상이 `ACTION_WINDOW_SEC`보다 짧으면 영상 전체를 1회
분류합니다.

**모델만 단독으로 확인하기**

```bash
python -m tools.verify_i3d                                   # 체크포인트 정보
python -m tools.verify_i3d --video sample.mp4                # 영상 전체 1회 분류
python -m tools.verify_i3d --video sample.mp4 --window-sec 30 --stride-sec 5   # 구간별 분류
python -m tools.verify_i3d --frames-dir some_clip_folder     # extract_multiclass.py 가 만든 클립 폴더
```

**알아둘 점 (학습 데이터 한계)**

- 체크포인트의 학습 데이터와 클래스 분포는 실제 매장 영상과 다를 수 있으므로
  배포 전에 현장 영상으로 점수와 오탐을 검증하세요.
- 기본 카테고리 임계값은 `0.60`이며 점수가 임계값보다 클 때 사건을 발급합니다.
  카테고리별 환경 변수로 조정할 수 있습니다.
- 재학습: `training/scripts/run_pipeline.sh` (원본 영상/라벨 XML 필요.
  이 패키지에는 포함되어 있지 않습니다).

## 설치

**Python 3.10 이상** 필요 (torch 요구사항). 가상환경 사용을 권장합니다.

```bash
# 가상환경 생성 및 활성화
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Mac / Linux
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

`requirements.txt`에는 `fastapi`, `uvicorn`, `ultralytics`(YOLO),
`opencv-python`, `numpy`, `pydantic`, `torch`, `torchvision`이 포함되어
있습니다. 최초 설치 시 다소 시간이 걸릴 수 있습니다. GPU를 쓰려면
`pip install -r requirements.txt` 이후 CUDA 버전에 맞는 `torch`를 별도로
재설치해야 합니다 (기본은 CPU 버전이 설치됩니다).

## 환경 변수 / 설정값 (`config/config.py`)

아래 값들은 모두 환경변수로 덮어쓸 수 있고, 지정하지 않으면 기본값이
사용됩니다.

---

환경변수 기본값 설명

---

`YOLO_MODEL_PATH` `yolo11n.pt` YOLO 가중치 경로
(FR-EVT-01)

`TRACKER_CONFIG` `bytetrack.yaml` ByteTrack 추적 설정
(FR-EVT-02)

`CONF_THRESHOLD` `0.25` 사람 탐지 최소 신뢰도

`AUTO_ANALYSIS_INPUT_DIR` `input/` 서버 시작 시 자동으로 감시할 영상 폴더

`AUTO_ANALYSIS_POLL_INTERVAL_SEC` `5.0` 입력 폴더 검색 간격(초)

`AUTO_ANALYSIS_DEFAULT_CAMERA_ID` `CAM-01` 파일명에서 카메라 ID를 찾지 못했을 때 사용할 값

`CLIP_MIN_SEC` / `CLIP_MAX_SEC` `2.0` / `4.0` 후보 클립 최소/최대
길이 (FR-EVT-04)

`EXPAND_X` / `EXPAND_Y` `0.50` / `0.50` crop 모드용 확장 설정. 현재 운영은
전체 화면 모드이므로 적용되지 않음

`EVENT_THRESHOLD` `0.60` 카테고리별 임계값을 따로 지정하지 않았을 때 적용할 기본값

`FALL_EVENT_THRESHOLD` / `LITTERING_EVENT_THRESHOLD` / `THEFT_EVENT_THRESHOLD` `fusion.json` 값 쓰러짐·쓰레기 투기·절도별 사건 기준(확률 환산값). 환경변수로 덮어쓸 수 있습니다

`EVENT_VIDEO_SEC` `10.0` 사건 영상 길이(초)
(FR-EVT-08)

`EVENT_VIDEO_PRE_SEC` / `5.0` / `5.0` 판정 시점 전/후 영상
`EVENT_VIDEO_POST_SEC` 길이. 후행 프레임 수신
뒤 확정

`CAMERA_DISCONNECT_SEC` `60.0` 카메라 끊김 판정 기준
시간(초) (FR-EVT-10)

`CONNECTION_CHECK_INTERVAL_SEC` `5.0` 등록된 카메라 끊김 확인 주기

`STREAM_RECONNECT_DELAY_SEC` `2.0` RTSP/HTTP 연결 실패 후 재시도 간격

`VLM_MAX_WORKERS` `2` VLM 비동기 검증 동시
실행 개수

`QWEN_VLM_BASE_URL` `http://127.0.0.1:8001/v1` vLLM OpenAI 호환 API
주소

`QWEN_VLM_MODEL` `Qwen/Qwen3-VL-8B-Instruct-FP8` 대표 이미지 3장 설명에
사용할 Qwen 모델

`QWEN_VLM_API_KEY` 비어 있음 vLLM을 `--api-key`로
실행한 경우에만 설정

`QWEN_VLM_TIMEOUT_SEC` `60.0` Qwen VLM 요청 제한
시간(초)

`QWEN_VLM_MAX_TOKENS` `512` 사건 설명 최대 출력
토큰 수

`QWEN_VLM_IMAGE_MAX_PIXELS` `147456` 이미지당 전송 최대 픽셀
수 (원본은 보존)

`QWEN_VLM_IMAGE_JPEG_QUALITY` `85` Qwen 전송용 JPEG 품질

`ALERT_WEBHOOK_URL` 비어 있음 사건 JSON을 POST할 1차
알림 웹훅

`VLM_RESULT_WEBHOOK_URL` 비어 있음 비동기 VLM 결과 콜백 주소

`I3D_WEIGHT_PATH` `training/runs/ours_a_final/best.pt` 행동분류 체크포인트(A,
화면 전체)(FR-EVT-05)

`I3D_CROP_WEIGHT_PATH` `training/runs/ours_c1_final/best.pt` 사람 크롭(C1) 체크포인트

`I3D_STACK_PATH` `training/runs/ours_stack/fusion.json` A+C1 동적 결합과 임계값

`I3D_USE_STACK` `1` 0이면 A 단독. C1·결합 파일이 없어도 A 단독으로 동작

`CROP_LARGEST_PERSON` `0` 1이면 C1 크롭을 가장 큰 사람 한 명으로 만듦

`VLM_ENABLED` `1` 0이면 VLM을 쓰지 않음(`QWEN_VLM_BASE_URL`을 비움)

`CONFIRMATIONS_REQUIRED` `1` 실시간 웹캠 모드에서 같은 비정상 클래스를 몇 번 연속
확인해야 사건으로 확정할지(이전 기본값 3)

`I3D_DEVICE` `auto` `auto` / `cpu` / `cuda`

`ACTION_WINDOW_SEC` / `4.0` / `2.0` 화면 전체 분류 창 및
`ACTION_WINDOW_OVERLAP_SEC` 겹침 길이

`I3D_BUFFER_FPS` `3.0` 분류용 프레임 버퍼 속도
(학습 영상이 3fps)

`EVENT_COOLDOWN_SEC` `30.0` 같은 카테고리 사건
재발급 금지 시간(초)

---

## 실제 점주 이미지 추론 흐름

점주는 이미지를 직접 업로드하지 않습니다. 행동 사건이 만들어지면 Path 1이
시작/중간/끝 대표 이미지 3장을 저장하고, vLLM에 비동기로 보내 관찰 내용·
불확실한 점·점주 확인사항을 생성합니다. 사건 저장과 1차 알림은 추론 완료를
기다리지 않습니다. React 사건 상세 화면은 `준비 중`에서 `완료` 또는 `실패`로
갱신되고, 완료 시 Qwen 모델명과 설명 3항목을 표시합니다.

### 서버 연결 확인

전체 Compose 실행에서는 `qwen-vllm` 서비스가 Qwen 모델 서버를 시작하며,
Docker Desktop의 WSL2 NVIDIA GPU 지원이 필요합니다. Compose를 사용하지 않고
WSL에서 직접 실행하려면 아래 **Qwen3-VL-8B-Instruct-FP8 + vLLM on WSL2 /
Ubuntu** 절차를 사용합니다. 두 방식 모두 호스트 포트 `8001`에서 모델을
확인합니다. Windows PowerShell에서 모델 ID를 확인합니다.

```powershell
(Invoke-RestMethod http://127.0.0.1:8001/v1/models).data.id
```

응답에 `Qwen/Qwen3-VL-8B-Instruct-FP8`가 있어야 합니다. 앱 기본값도 이 모델 ID와
일치합니다. Docker Compose의 탐지기는 Docker 네트워크 내부 주소
`http://qwen-vllm:8000/v1`을 사용하며, 호스트에서 직접 확인할 때는
`http://127.0.0.1:8001/v1`을 사용합니다. 다른 VLM 서버를 쓰려면
`docker-compose.yml`의 `storeops-ai` 서비스에 설정된 `QWEN_VLM_BASE_URL`을
수정한 뒤 탐지 컨테이너를 재생성합니다. Compose의 필수 토큰, 모델 파일과
전체 실행 절차는 [루트 README](../README.md)를 확인하세요.

### 대표 이미지로 사전 점검

실제 사건에서 저장된 `output/representative_images/E001_1.jpg`부터 `_3.jpg`까지
세 장을 사용해 앱에 사건을 넣기 전에 추론을 확인할 수 있습니다. 프로젝트 루트의
PowerShell에서 실행합니다.

```powershell
Push-Location .\storeops_ai
python -m tools.verify_vlm `
  --images .\output\representative_images\E001_1.jpg `
           .\output\representative_images\E001_2.jpg `
           .\output\representative_images\E001_3.jpg `
  --event-id VLM-TEST --camera-id CAM-01
Pop-Location
```

`E001`은 실제로 존재하는 사건 ID로 바꿉니다. 성공하면 JSON 설명과 모델명이 출력됩니다.
실패하면 URL·모델 ID·이미지 파일·vLLM 로그를 확인합니다.

### 점주 사용

1. 루트 README의 필수 환경변수와 모델 파일을 준비합니다.
2. `docker compose up -d --build`로 StoreOps 서비스를 실행합니다.
3. vLLM `/v1/models`가 정상 응답하는지 확인합니다.
4. 점주가 `http://localhost:5173`에 로그인하고 사건을 확인합니다.
5. 행동 사건이 발생하면 대표 이미지가 자동 추론되고, 상세 화면에 결과가 붙습니다.

카메라 끊김 사건은 이미지가 없으므로 VLM을 실행하지 않습니다. vLLM이 꺼져 있거나
추론에 실패해도 사건과 1차 알림은 유지되고, 상세 화면에 실패 상태가 표시됩니다.
단, 현재 Compose에서는 탐지 서비스가 Qwen VLM의 healthy 상태에 의존하므로
Compose 전체를 시작하려면 Qwen 서버가 준비되어야 합니다. VLM 없이 탐지 API만
실행할 때는 로컬 FastAPI 실행 절차를 사용하세요.

예시 (Path 1에서 사람 탐지 신뢰도 임계값을 바꾸고 싶을 때):

```powershell
cd storeops_ai
$env:CONF_THRESHOLD="0.4"
python -m pipelines.path1_behavior --video .\input\sample.mp4
```

Linux/WSL에서는 서비스 폴더에서 다음처럼 실행합니다.

```bash
CONF_THRESHOLD=0.4 python -m pipelines.path1_behavior --video ./input/sample.mp4
```

## Path 1 테스트

CLI로 영상 파일 하나를 직접 분석합니다. 프로젝트 루트에서 실행하세요.

**인자**

---

인자 필수 설명

---

`--video` ✅ 분석할 영상 파일 경로

`--camera-id` ❌ (기본 `CAM001`) 카메라 식별자

`--i3d-weight` ❌ I3D 가중치 경로 (생략하면
`training/runs/ours_a_final/best.pt`)

`--debug` ❌ 디버그 로그 출력

---

**실행 예시**

```powershell
cd storeops_ai
python -m pipelines.path1_behavior --video "input\sample.mp4" --camera-id CAM-01 --debug
```

결과는 표준출력에 JSON 배열(`EventCandidate` 리스트)로 출력되고, 동시에
`output/` 아래에 클립/이미지/이벤트 JSON이 저장됩니다.

> 분류 로그(`[I3D] t=..s 카테고리 confidence | 점수`)가 분류 1회마다
> 출력됩니다. 임계값 미만이거나 '정상'이면 사건이 발급되지 않습니다.
> 가중치 파일이 없으면(`I3D_WEIGHT_PATH` 잘못 지정 등) 탐지/추적까지만
> 수행하고 사건은 발급하지 않습니다.

API로도 호출할 수 있습니다: `POST /path1/analyze`
(`{"video_path": "input/sample.mp4", "camera_id": "CAM-01"}`). 서버는
`input/` 폴더에 안정적으로 기록된 `.mp4`, `.avi`, `.mov`, `.mkv` 파일도 자동
분석합니다. 파일명이 `CAM-01_...`처럼 `CAM` 접두사로 시작하면 그 접두부를
카메라 ID로 사용하고, 그렇지 않으면 `AUTO_ANALYSIS_DEFAULT_CAMERA_ID`를
사용합니다. 처리된 파일과 실패 파일은 각각 `input/processed/`, `input/failed/`로
이동합니다.

## Path 2 테스트

FastAPI 서버를 띄워서 API 호출로 테스트합니다.

**서버 실행**

```powershell
cd storeops_ai
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

로컬 기본 포트는 `8000`이며 Docker Compose에서는 호스트 `8100` 포트로
연결됩니다. API 문서는 로컬 실행 시
[http://localhost:8000/docs](http://localhost:8000/docs)에서 확인합니다.

**엔드포인트**

---

| 메서드      | 경로                                      | 설명                                 |
| ----------- | ----------------------------------------- | ------------------------------------ |
| `GET`       | `/`                                       | 서비스 상태                          |
| `POST`      | `/path1/analyze`                          | 지정한 영상 동기 분석                |
| `GET`       | `/path1/events`                           | 로컬 이벤트 JSON 목록                |
| `GET`       | `/path1/events/{event_id}`                | 이벤트 상세                          |
| `GET`       | `/path1/events/{event_id}/clip`           | 사건 영상 다운로드                   |
| `GET`       | `/path1/events/{event_id}/images/{index}` | 대표 이미지 다운로드(0부터 시작)     |
| `PATCH`     | `/path1/events/{event_id}/status`         | 상태를 `확인` 또는 `오탐`으로 변경   |
| `WebSocket` | `/path1/ws/events`                        | 신규·변경 이벤트 실시간 전달         |
| `POST`      | `/path2/frame`                            | 외부 수신기가 프레임 수신 시각 갱신  |
| `POST`      | `/path2/check`                            | 연결 끊김 여부 수동 확인             |
| `GET`       | `/path2/state/{camera_id}`                | 현재 연결 및 스트림 상태 조회        |
| `POST`      | `/path2/streams`                          | RTSP/HTTP 프레임 수신 워커 시작·갱신 |
| `DELETE`    | `/path2/streams/{camera_id}`              | 해당 카메라 수신 워커 중지           |

---

**React 실시간 메시지 형식**

WebSocket 연결 주소는 `ws://서버주소:8000/path1/ws/events`이다. 연결 직후
`connection.ready`를 보내고, 이후 아래 두 종류의 이벤트를 전달한다.
`event`에는 사건 JSON 전체가 들어간다.

```json
{"type":"event.created","occurred_at":"2026-09-22T...Z","event":{"event_id":"E015","status":"사건후보"}}
{"type":"event.updated","occurred_at":"2026-09-22T...Z","event":{"event_id":"E015","status":"확인"}}
```

점주 상태 변경 요청 예시는 다음과 같다.

```json
PATCH /path1/events/E015/status
{"status":"확인"}
```

`CORS_ALLOW_ORIGINS` 환경변수에 React 개발/운영 URL을 쉼표로 구분해
지정한다. 이 서비스의 REST API와 WebSocket에는 사용자 인증이 없으므로
신뢰할 수 있는 네트워크에서만 사용하고 인터넷에 직접 노출하지 않는다.

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

`timestamp`(unix epoch, 초) 필드를 요청 본문에 함께 보내면 실제 시간
대신 해당 시각 기준으로 테스트할 수 있습니다 (자동화 테스트 시 유용).

## 출력

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

사건 ID는 `output/events/` 안의 기존 `E***.json` 파일 중 가장 큰 번호
다음 번호로 자동 채번됩니다 (`E999` 다음은 `E1000`). 카메라 끊김 사건은
`event_video_path`, `representative_images`, `scores`가 비워진 채로
`disconnect_seconds` / `disconnect_threshold_seconds`만 채워져
저장됩니다.

## Windows 로컬 실행

저장소 루트에서 PowerShell을 열고 CCTV 서비스 가상환경을 만듭니다.

```powershell
cd storeops_ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 문서는 `http://localhost:8000/docs`입니다. 기본 체크포인트가 없으면
행동 분류와 이벤트 발급은 비활성화됩니다. YOLO 가중치와 학습 체크포인트는
각각 `YOLO_MODEL_PATH`, `I3D_WEIGHT_PATH`가 가리키는 경로에 준비하세요.
Compose 실행과 백엔드 연동은 [루트 README](../README.md)를 참고하세요.

## 트러블슈팅

---

증상 원인 / 해결

---

`I3D 가중치 없음 ... 행동분류 비활성 상태` `training/runs/ours_a_final/best.pt`
경고 가 없거나 `I3D_WEIGHT_PATH` 가
틀렸습니다.

영상이 길면 메모리 부족 Path1은 분석 중 모든 원본/주석
프레임을 메모리에 보관합니다 (기존
구조). 긴 영상은 잘라서 분석하세요.

`pip install` 이 매우 오래 걸림 / 용량이 큼 `ultralytics` 설치 시 `torch`가
함께 설치되기 때문입니다 (정상).
네트워크 상황에 따라 수 분 걸릴 수
있습니다.

YOLO 실행 시 `bytetrack.yaml` 관련 오류 `ultralytics` 패키지 내장 기본
설정을 사용하므로 별도 파일 생성
없이 그대로 실행하면 됩니다.
커스텀하려면 동일한 이름의 yaml을
프로젝트에 두고 경로를 맞추세요.

`output/` 폴더에 아무 것도 생성되지 않음 Path1은 사건이 하나도 발급되지
않으면 이벤트 JSON/클립/이미지를
만들지 않습니다. 분류 로그의
카테고리·점수를 확인하세요
(정상으로 분류되었거나 임계값
미만). 로그 자체가 없으면 YOLO가
사람을 탐지하지 못한 경우입니다
(`CONF_THRESHOLD`를 낮춰보세요).

Path2 끊김 사건이 계속 재생성됨 `/path2/frame`을 호출해 정상 수신
상태로 되돌리기 전까지는
`event_created_for_disconnect`
플래그로 중복 생성을 막습니다. 계속
생성된다면 매 호출마다 새로운
`CameraConnectionMonitor`
인스턴스가 만들어지고 있는 건
아닌지(서버 재시작 등) 확인하세요.
현재 구현은 인메모리 상태이므로
서버를 재시작하면 상태가
초기화됩니다.

---

---

## FR-EVT Path 1 실시간 웹캠 테스트

학습된 `training/runs/ours_a_final/best.pt`(+ C1·결합)를 실제 Path 1에 연결하기 전에,
먼저 S3D 행동분류 모델 자체가 웹캠 영상에서 정상적으로 반응하는지 확인할
수 있습니다.

이 프로젝트에는 학습 완료된 S3D 행동분류 모델이 포함되어 있습니다.

기본 모델: `training/runs/ours_a_final/best.pt`(A) + `ours_c1_final/best.pt`(C1) + `ours_stack/fusion.json`

운영 화면에 표시하는 점수는 `NORMAL`, `FALL`, `LITTERING`, `THEFT`입니다.
싸움·파손·방화는 학습하지 않아 표시하지 않습니다.

## 웹캠 실행

```powershell
cd storeops_ai
.\.venv\Scripts\Activate.ps1
python webcam_i3d.py
```

웹캠 번호가 0이 아니면 `--camera 1`을 지정합니다. 기본 입력 창은 학습과 같은 4초이며,
2초마다 추론합니다. CPU가 느리면 `--interval-sec 4`처럼 간격을 늘리거나
`--window-sec 10`으로 입력 창을 조정할 수 있습니다. 이 도구는 모델 점검 전용이며
이벤트를 발급하거나 백엔드로 전송하지 않습니다.

---

# Qwen3-VL-8B-Instruct-FP8 + vLLM on WSL2 / Ubuntu

이 절에서는 Windows의 WSL2 Ubuntu에 NVIDIA GPU용 Python 환경을 구성하고,
Qwen3-VL-8B-Instruct-FP8을 vLLM 서버로 실행하는 방법을 안내합니다.

## 1. 최종 구성

```text
Windows
└─ WSL2
  └─ Ubuntu 26.04.1 LTS
    └─ Python virtual environment: qwen-vllm
      ├─ PyTorch 2.13.0+cu130
      ├─ FlashInfer 0.6.18.post1
      └─ vLLM
        └─ Qwen/Qwen3-VL-8B-Instruct-FP8
```

### 실제 확인된 하드웨어 / 소프트웨어

항목 값

---

OS Host Windows
Linux 환경 WSL2
Ubuntu 26.04.1 LTS (Resolute Raccoon)
Architecture `x86_64`
GPU NVIDIA RTX A4000
GPU VRAM 16 GB
PyTorch `2.13.0+cu130`
PyTorch CUDA `13.0`
CUDA available `True`
FlashInfer `0.6.18.post1`
Model `Qwen/Qwen3-VL-8B-Instruct-FP8`
vLLM Port `8001`
Max model length `4096`

---

# 2. Windows PowerShell에서 WSL2 설치

## 2.1 PowerShell을 관리자 권한으로 실행

Windows 시작 메뉴에서:

```text
PowerShell
```

을 검색한 다음 **관리자 권한으로 실행**합니다.

다음 명령으로 WSL을 설치합니다.

```powershell
wsl --install
```

설치가 끝나면 Windows를 재부팅합니다.

---

## 2.2 WSL 상태 확인

재부팅 후 PowerShell을 다시 실행합니다.

```powershell
wsl --status
```

또는:

```powershell
wsl -l -v
```

정상적으로 WSL2가 사용되고 있는지 확인합니다.

예:

```text
NAME      STATE           VERSION
Ubuntu    Running         2
```

`VERSION`이 `2`인지 확인합니다.

---

# 3. Ubuntu 실행

PowerShell에서:

```powershell
wsl
```

또는 설치된 Ubuntu가 있다면:

```powershell
ubuntu
```

Ubuntu 터미널에 들어갑니다.

현재 사용 중인 Linux 배포판을 확인합니다.

```bash
cat /etc/os-release
```

이번에 실제 사용한 환경은:

```text
PRETTY_NAME="Ubuntu 26.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="26.04"
VERSION="26.04.1 LTS (Resolute Raccoon)"
UBUNTU_CODENAME=resolute
```

CPU 아키텍처도 확인합니다.

```bash
uname -m
```

정상 결과:

```text
x86_64
```

---

# 4. Windows NVIDIA 드라이버 확인

WSL에서 NVIDIA GPU를 사용하려면 **Windows에 NVIDIA 드라이버가 설치되어
있어야 합니다.**

중요:

> WSL Ubuntu 안에 일반적인 Linux용 NVIDIA 그래픽 드라이버를 별도로
> 설치하지 않습니다.

Windows에서 NVIDIA 드라이버가 정상 설치되어 있다면 Ubuntu에서 다음
명령으로 확인할 수 있습니다.

```bash
nvidia-smi
```

이번 환경에서는 다음 GPU가 확인되었습니다.

```text
NVIDIA RTX A4000
```

VRAM:

```text
16376 MiB
```

약 16 GB입니다.

`nvidia-smi`가 정상적으로 GPU 정보를 출력하면 다음 단계로 진행합니다.

---

# 5. Ubuntu 패키지 업데이트

Ubuntu에서:

```bash
sudo apt update
sudo apt upgrade -y
```

기본 개발 도구도 설치합니다.

```bash
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    software-properties-common
```

---

# 6. Python 확인

Python 버전을 확인합니다.

```bash
python3 --version
```

이번 환경에서는 Python 3.14 계열 환경에서 vLLM이 실행되었습니다.

Python 실행 파일도 확인할 수 있습니다.

```bash
which python3
```

---

# 7. Qwen용 Python 가상환경 생성

Ubuntu에 `ensurepip`가 없는 경우가 있어 `uv`로 사용자 영역에 Python과 가상환경을 설치합니다. `sudo` 권한은 필요하지 않습니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.14
uv venv "$HOME/qwen-vllm" --python 3.14
source "$HOME/qwen-vllm/bin/activate"
```

가상환경의 Python 경로를 확인합니다.

```bash
"$HOME/qwen-vllm/bin/python" --version
```

---

# 8. CUDA compiler / linker 준비

vLLM/FlashInfer는 첫 실행에서 CUDA 커널을 JIT 컴파일합니다. 아래 패키지들은
가상환경 안에 설치되며, 시스템 CUDA Toolkit 전체를 별도로 설치하지 않습니다.

```bash
source "$HOME/.local/bin/env"
uv pip install --python "$HOME/qwen-vllm/bin/python" \
  nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88
"$HOME/qwen-vllm/lib/python3.14/site-packages/nvidia/cu13/bin/nvcc" --version
```

`tools/start_qwen_vllm.sh`가 가상환경 안의 CUDA compiler, runtime library, WSL의
GPU driver library 경로를 설정합니다. 따라서 서버 시작에는 이 스크립트를 사용합니다.

---

# 9. PyTorch 설치 및 CUDA 확인

가상환경이 활성화된 상태에서:

```bash
uv pip install --python "$HOME/qwen-vllm/bin/python" --upgrade pip
```

설치된 PyTorch를 확인합니다.

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

이번에 실제 확인된 결과:

```text
torch: 2.13.0+cu130
torch CUDA: 13.0
CUDA available: True
```

GPU 이름까지 확인:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

예상:

```text
NVIDIA RTX A4000
```

---

# 10. FlashInfer 확인

설치된 FlashInfer 버전을 확인합니다.

```bash
python -c "import flashinfer; print('flashinfer:', flashinfer.__version__)"
```

이번에 실제 확인된 버전:

```text
flashinfer: 0.6.18.post1
```

---

# 11. vLLM 설치

가상환경이 활성화되어 있는지 확인합니다.

```bash
which python
```

다음처럼 나와야 합니다.

```text
/home/user/qwen-vllm/bin/python
```

vLLM을 설치합니다.

```bash
uv pip install --python "$HOME/qwen-vllm/bin/python" vllm
```

설치 확인:

```bash
vllm --version
```

---

# 12. 사용할 Qwen 모델

RTX A4000의 VRAM은 약 16 GB이므로 다음 FP8 모델을 사용합니다.

```text
Qwen/Qwen3-VL-8B-Instruct-FP8
```

---

# 13. vLLM 서버 실행

Windows 드라이브가 `/mnt/c`에 연결된 WSL Ubuntu 터미널에서 저장소 루트로 이동한 뒤 실행합니다.

```bash
cd /mnt/c/Users/user/StroeOpsAI
```

서버 실행:

```bash
bash storeops_ai/tools/start_qwen_vllm.sh
```

이 스크립트가 가상환경의 CUDA 13 compiler 및 링크 경로를 설정하고, 모델
`Qwen/Qwen3-VL-8B-Instruct-FP8`, 포트 `8001`, 요청당 이미지 3장, 최대 문맥
길이 `4096`으로 서버를 실행합니다.

### 스크립트 설정값

| 항목                       | 값                              | 설명                                  |
| -------------------------- | ------------------------------- | ------------------------------------- |
| `QWEN_VLM_MODEL`           | `Qwen/Qwen3-VL-8B-Instruct-FP8` | 모델 ID 환경변수                      |
| `QWEN_VLLM_PORT`           | `8001`                          | 공개할 API 포트 환경변수              |
| `--host`                   | `0.0.0.0`                       | 스크립트에서 고정                     |
| `--limit-mm-per-prompt`    | `{"image":3,"video":0}`         | 요청당 이미지 최대 3장, 비디오 미사용 |
| `--gpu-memory-utilization` | `0.90`                          | 스크립트에서 고정                     |
| `--max-model-len`          | `4096`                          | 스크립트에서 고정                     |

# 14. 서버 정상 실행 확인

서버 터미널에서 다음 메시지가 나오면 API 서버가 시작된 것입니다.

```text
Application startup complete.
```

실제 환경에서도 다음 메시지를 확인했습니다.

```text
Application startup complete.
```

---

# 15. `/v1/models` API 확인

**서버를 실행 중인 터미널은 그대로 둡니다.**

새 Ubuntu/WSL 터미널을 열고:

```bash
source ~/qwen-vllm/bin/activate
```

다음 명령을 실행합니다.

```bash
curl http://127.0.0.1:8001/v1/models
```

정상적으로 실행되면 HTTP `200 OK`가 반환됩니다.

실제 환경에서:

```text
GET /v1/models HTTP/1.1" 200 OK
```

를 확인했습니다.

응답에서 다음 모델이 확인됩니다.

```json
{
  "root": "Qwen/Qwen3-VL-8B-Instruct-FP8",
  "max_model_len": 4096
}
```

---

# 16. 최종 정상 상태

현재까지 실제로 확인된 상태:

```text
Windows
  ↓
WSL2
  ↓
Ubuntu 26.04.1 LTS
  ↓
x86_64
  ↓
NVIDIA RTX A4000 16GB
  ↓
Python virtual environment: qwen-vllm
  ↓
PyTorch 2.13.0+cu130
  ↓
CUDA available: True
  ↓
FlashInfer 0.6.18.post1
  ↓
vLLM
  ↓
Qwen/Qwen3-VL-8B-Instruct-FP8
  ↓
API Server :8001
  ↓
GET /v1/models → 200 OK
```

서버 실행 확인:

```text
Application startup complete.
```

API 확인:

```text
GET /v1/models HTTP/1.1" 200 OK
```

확인된 모델:

```text
Qwen/Qwen3-VL-8B-Instruct-FP8
```

확인된 최대 context:

```text
4096
```

---

# 17. 서버 재실행 시 필요한 최소 명령

이미 모든 설치가 끝난 상태라면 이후에는 다음만 실행하면 됩니다.

## 터미널 1

```bash
bash /mnt/c/Users/user/StroeOpsAI/storeops_ai/tools/start_qwen_vllm.sh
```

## 터미널 2

```bash
curl http://127.0.0.1:8001/v1/models
```

정상 응답:

```text
200 OK
```

---

# 18. 주요 확인 명령 모음

### WSL / Ubuntu

```bash
cat /etc/os-release
uname -m
```

### GPU

```bash
nvidia-smi
```

### CUDA Compiler

```bash
which nvcc
nvcc --version
```

### Python

```bash
python --version
which python
```

### PyTorch

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

### FlashInfer

```bash
python -c "import flashinfer; print('flashinfer:', flashinfer.__version__)"
```

### vLLM

```bash
vllm --version
```

### API

```bash
curl http://127.0.0.1:8001/v1/models
```

---

## 참고

이 README의 실행 결과와 버전 정보는 현재 구성에서 실제로 확인된 값을
기준으로 작성했습니다.

특히 다음 항목은 실제 정상 동작이 확인되었습니다.

- Ubuntu 26.04.1 LTS / x86_64
- NVIDIA RTX A4000 16GB
- PyTorch `2.13.0+cu130`
- CUDA available `True`
- FlashInfer `0.6.18.post1`
- Qwen `Qwen/Qwen3-VL-8B-Instruct-FP8`
- vLLM API 서버 정상 기동
- `/v1/models` HTTP `200 OK`
- 대표 이미지 3장 실제 추론 및 E001 점주 화면 callback 성공
- `max_model_len=4096`

## 기여

모델·임계값·파이프라인을 변경하면 관련 영상 또는 모델 단독 검증을 실행하고, 필요한 경우 Compose 연동 상태도 확인합니다. 프로젝트 전체 작업·브랜치·커밋 규칙은 [루트 지침](../AGENTS.md)을 참고하세요.

## 라이선스

현재 저장소에는 별도 라이선스가 지정되어 있지 않습니다.
