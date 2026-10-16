"""FR-EVT 공통 설정.

현재 배포 가중치는 torchvision S3D이다. I3D로 교체하려면 I3D 구조로 재학습한
체크포인트가 필요하므로, 서비스 요구사항은 화면 전체 temporal 행동분류로 정의한다.
"""
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# input 폴더에 새 영상이 들어오면 사람이 API를 호출하지 않아도 자동으로 분석한다.
INPUT_DIR = Path(os.getenv("AUTO_ANALYSIS_INPUT_DIR", str(BASE_DIR / "input")))
AUTO_ANALYSIS_POLL_INTERVAL_SEC = float(os.getenv("AUTO_ANALYSIS_POLL_INTERVAL_SEC", "5.0"))
AUTO_ANALYSIS_DEFAULT_CAMERA_ID = os.getenv("AUTO_ANALYSIS_DEFAULT_CAMERA_ID", "CAM-01")


# FR-EVT-01~02
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11s-pose.pt")   # 사람 박스·추적·관절을 한 번에(2026-10-15 사용자 결정). 이전 YOLO11n은 환경변수로 되돌릴 수 있다
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
PERSON_CLASS_ID = 0

# FR-EVT-03~04
CLIP_MIN_SEC = float(os.getenv("CLIP_MIN_SEC", "2.0"))
CLIP_MAX_SEC = float(os.getenv("CLIP_MAX_SEC", "4.0"))
# 증거 보존용 bbox 확장 비율(확장 크롭을 쓰는 카테고리가 있을 때만 사용; 현재 학습한 카테고리에는 없다).
EXPAND_X = float(os.getenv("EXPAND_X", "0.50"))
EXPAND_Y = float(os.getenv("EXPAND_Y", "0.50"))
EXPANDED_CROP_CATEGORIES: set = set()

# FR-EVT-05: 우리가 학습한 4개(정상·전도·유기·절도)만 운영 점수로 노출한다. 싸움·파손·방화는 학습하지 않아 뺐다.
CATEGORIES = ["정상", "쓰러짐", "쓰레기 투기", "절도"]
# 임시 기준값은 비정상 클래스 모두 0.60으로 두고 환경변수로 조정할 수 있다.
EVENT_THRESHOLD = float(os.getenv("EVENT_THRESHOLD", "0.60"))

def _category_threshold(env_name: str, default: str) -> float:
    """개별 설정이 없으면 기존 공통 EVENT_THRESHOLD 설정을 계속 존중한다."""
    return float(os.getenv(env_name, os.getenv("EVENT_THRESHOLD", default)))

I3D_CROP_WEIGHT_PATH = os.getenv("I3D_CROP_WEIGHT_PATH", str(BASE_DIR / "training" / "runs" / "ours_c1_final" / "best.pt"))   # 사람 크롭(C1) 모델
I3D_STACK_PATH = os.getenv("I3D_STACK_PATH", str(BASE_DIR / "training" / "runs" / "ours_stack" / "fusion.json"))                # A+C1 동적 결합·임계값(OOF). 이전 로지스틱은 stack_logistic_old.json
# 1이면 C1 크롭을 가장 큰 인물 한 명의 박스로만 만든다(MVP 단순화). 기본 0 = 창 안 모든 사람 박스의 합집합(학습 때 규칙). 사람이 여럿인 창에서 1이면 C1 정확도가 97%→80%로 떨어진다(실측).
CROP_LARGEST_PERSON = os.getenv("CROP_LARGEST_PERSON", "0") == "1"
I3D_USE_STACK = os.getenv("I3D_USE_STACK", "1") == "1" and Path(I3D_CROP_WEIGHT_PATH).is_file() and Path(I3D_STACK_PATH).is_file()

def _stack_default(category: str, fallback: str) -> str:
    """스택을 쓰면 OOF에서 정한 임계값(확률)을, 아니면 A 단독 기본값을 쓴다. 환경변수가 있으면 그것이 우선."""
    if I3D_USE_STACK:
        import json
        spec = json.loads(Path(I3D_STACK_PATH).read_text(encoding="utf-8"))["classes"].get(category)
        if spec:
            return str(1 / (1 + math.exp(-spec["threshold_logit"])))
    return fallback

# A 단독 기본값 = OOF(재현율>=0.9, 오경보 최소)에서 고른 점수 6.23/4.70/2.15의 확률값(sigmoid). 스택 기본값은 stack.json의 OOF 임계값.
CATEGORY_THRESHOLDS = {
    "쓰러짐": _category_threshold("FALL_EVENT_THRESHOLD", _stack_default("쓰러짐", "0.998")),
    "쓰레기 투기": _category_threshold("LITTERING_EVENT_THRESHOLD", _stack_default("쓰레기 투기", "0.991")),
    "절도": _category_threshold("THEFT_EVENT_THRESHOLD", _stack_default("절도", "0.896")),
}

# FR-EVT-05: 학습된 행동분류 모델 (torchvision S3D, training/train_i3d.py 로 학습)
I3D_WEIGHT_PATH = os.getenv("I3D_WEIGHT_PATH", str(BASE_DIR / "training" / "runs" / "ours_a_final" / "best.pt"))
I3D_DEVICE = os.getenv("I3D_DEVICE", "auto")  # auto | cpu | cuda

# 학습 라벨(영문) -> 서비스 카테고리(한글). 기존 체크포인트의 미요구 클래스는 점수 응답에서 제외한다.
MODEL_LABEL_TO_CATEGORY = {
    "normal": "정상", "fall": "쓰러짐", "broken": "파손",
    "abandon": "쓰레기 투기", "fight": "싸움",
    "fire": "방화", "theft": "절도",
}

# 화면 전체 입력만 운영한다. 기존 I3D_* 환경변수 이름은 하위 호환을 위해 유지한다.
ACTION_INPUT_MODE = os.getenv("ACTION_INPUT_MODE", "full")
I3D_INPUT_MODE = ACTION_INPUT_MODE
# 4초 창을 2초마다 실행한다. 즉 인접 창은 2초(50%) 겹친다.
I3D_WINDOW_SEC = float(os.getenv("ACTION_WINDOW_SEC", os.getenv("I3D_WINDOW_SEC", "4.0")))
I3D_WINDOW_OVERLAP_SEC = float(os.getenv("ACTION_WINDOW_OVERLAP_SEC", "2.0"))
I3D_INFER_INTERVAL_SEC = I3D_WINDOW_SEC - I3D_WINDOW_OVERLAP_SEC
if I3D_INFER_INTERVAL_SEC <= 0:
    raise ValueError("ACTION_WINDOW_OVERLAP_SEC 는 ACTION_WINDOW_SEC 보다 작아야 합니다.")
# 학습 영상이 3fps였으므로 분류용 버퍼도 이 속도로 쌓는다.
I3D_BUFFER_FPS = float(os.getenv("I3D_BUFFER_FPS", "3.0"))
# 같은 카테고리 사건 재발급 금지 시간(초). 윈도우가 30초면 같은 사건이 여러 번 윈도우에 걸린다.
# 행동 판정에 쓸 사람: 전신이 보이는 사람만(화면 가장자리에 안 닿음), 많이 겹친 박스는 한 명, 큰 순서로 상위 K명(사용자 결정 2026-10-15).
PERSON_TOP_K = int(os.getenv("PERSON_TOP_K", "3"))
PERSON_MIN_CONF = float(os.getenv("PERSON_MIN_CONF", "0.6"))          # BoT-SORT 논문 기본 검출 임계값 τ=0.6. 선택 단계에만 쓴다(C1 입력은 학습과 같은 규칙으로 모든 사람 박스를 쓴다)
PERSON_MERGE_IOU = float(os.getenv("PERSON_MERGE_IOU", "0.7"))        # YOLO NMS 기본 IoU 0.7(울트라리틱스). 직접 근거는 약함 — 재검토
PERSON_EDGE_MARGIN = float(os.getenv("PERSON_EDGE_MARGIN", "0.01"))   # 논문에 없음 — 재검토
# 새로 멈춘 물건 영역 확인(유기). Porikli 외 2008 이중 전경. 근거 논문 값은 행동인식_분석.md 참조.
STATIC_OBJECT_ENABLED = os.getenv("STATIC_OBJECT_ENABLED", "1") == "1"
STATIC_ABANDONED_CATEGORIES = set(os.getenv("STATIC_ABANDONED_CATEGORIES", "쓰레기 투기").split(","))   # 새로 놓인 물건이 있어야 경보
STATIC_REMOVED_CATEGORIES = set(os.getenv("STATIC_REMOVED_CATEGORIES", "절도").split(","))               # 물건이 없어져야 경보
STATIC_SCALE = float(os.getenv("STATIC_SCALE", "0.25"))                      # 논문: 4분의 1 해상도
STATIC_PROC_FPS = float(os.getenv("STATIC_PROC_FPS", "5.0"))                 # 논문: 짧은 배경 5fps
STATIC_LONG_PERIOD_SEC = float(os.getenv("STATIC_LONG_PERIOD_SEC", "5.0"))   # 논문: 긴 배경 5초마다 갱신
STATIC_RATIO = float(os.getenv("STATIC_RATIO", "30"))                        # 논문: 짧은 배경 학습률 = 긴 배경의 30배
STATIC_MAX_EVIDENCE = int(os.getenv("STATIC_MAX_EVIDENCE", "50"))            # 논문 범위 [50,500]프레임, 전형 300(30fps에서 약 10초)을 5fps 처리 기준 10초로 환산
STATIC_DECAY =float(os.getenv("STATIC_DECAY", "1"))                         # 논문 실험값 k=1
STATIC_LONG_ALPHA = float(os.getenv("STATIC_LONG_ALPHA", "0.002"))           # 논문에 없음 — 측정으로 재검토
STATIC_MIN_AREA_FRAC = float(os.getenv("STATIC_MIN_AREA_FRAC", "0.0005"))    # 논문에 없음 — 측정으로 재검토
STATIC_HOLD_SEC = float(os.getenv("STATIC_HOLD_SEC", "30.0"))                # 논문에 없음 — 측정으로 재검토
STATIC_OWNER_SEC = float(os.getenv("STATIC_OWNER_SEC", "20.0"))              # 물건 영역 근처에 사람이 있었는지 볼 과거 시간. 논문에 없음 — 측정으로 재검토
STATIC_OWNER_RADIUS = float(os.getenv("STATIC_OWNER_RADIUS", "0.2"))        # 물건 영역이 손목에서 사람 키의 몇 배 안이면 그 사람 것으로 보나. 임시값(0.3에서 엄격하게 줄임, 2026-10-15). 논문에 수치 없음 — 로그의 거리 분포로 재검토
STATIC_WRIST_CONF = float(os.getenv("STATIC_WRIST_CONF", "0.3"))            # 손목 관절을 쓸 최소 신뢰도. 논문에 없음 — 재검토
STATIC_DEPART_SEC = float(os.getenv("STATIC_DEPART_SEC", "2.0"))            # 그 번호가 이만큼 안 보이면 사라진 것(떠남)으로 본다. 논문에 없음 — 재검토
STATIC_PENDING_SEC = float(os.getenv("STATIC_PENDING_SEC", "30.0"))          # 모델이 먼저 이상으로 본 뒤 물건 확인을 기다리는 시간. 논문에 없음 — 측정으로 재검토
STATIC_PERSON_PAD = float(os.getenv("STATIC_PERSON_PAD", "0.1"))             # 논문에 없음(사람 박스 제외는 우리 추가)
STATIC_GROW_TOL = int(os.getenv("STATIC_GROW_TOL", "10"))                    # Tian 외 2011에 수치 없음 — 측정으로 재검토
STATIC_SIMILAR_RATIO = float(os.getenv("STATIC_SIMILAR_RATIO", "1.5"))       # Tian 외 2011에 수치 없음 — 측정으로 재검토
EVENT_COOLDOWN_SEC = float(os.getenv("EVENT_COOLDOWN_SEC", "30.0"))

# FR-EVT-08
EVENT_VIDEO_SEC = float(os.getenv("EVENT_VIDEO_SEC", "10.0"))
# 사건 판정 시점 기준 앞 5초 + 뒤 5초. 끝 프레임은 실시간으로 도착한 뒤 영상을 확정한다.
EVENT_VIDEO_PRE_SEC = float(os.getenv("EVENT_VIDEO_PRE_SEC", "5.0"))
EVENT_VIDEO_POST_SEC = float(os.getenv("EVENT_VIDEO_POST_SEC", "5.0"))
if abs(EVENT_VIDEO_PRE_SEC + EVENT_VIDEO_POST_SEC - EVENT_VIDEO_SEC) > 1e-9:
    raise ValueError("EVENT_VIDEO_PRE_SEC + EVENT_VIDEO_POST_SEC 는 EVENT_VIDEO_SEC 와 같아야 합니다.")

# FR-EVT-09~12
CAMERA_DISCONNECT_SEC = float(os.getenv("CAMERA_DISCONNECT_SEC", "60.0"))

# 저장 경로 (Path 객체여야 pipelines/path1_behavior.py 의 `CLIP_DIR / "..."` 가 동작한다)
CLIP_DIR = BASE_DIR / "output" / "clips"
EVENT_DIR = BASE_DIR / "output" / "events"
IMAGE_DIR = BASE_DIR / "output" / "representative_images"

# FR-EVT-15
VLM_MAX_WORKERS = int(os.getenv("VLM_MAX_WORKERS", "2"))
# VLM_ENABLED=0이면 VLM을 쓰지 않는다(주소를 비움). Windows에서는 환경변수를 빈 값으로 둘 수 없어(set 변수= 는 변수를 지운다) 별도 스위치를 둔다.
# 사건이 난 순간에 VLM이 대표 이미지를 읽게 할지 여부. 기본 0 = 끔(사용자 결정 2026-10-08):
# 경보에는 사건 종류·시각·카메라와 클립·이미지만 보내고, VLM 해석은 점주가 지난 사건을 물을 때만 질문 시점에 한다.
VLM_ON_EVENT = os.getenv("VLM_ON_EVENT", "0") == "1"
QWEN_VLM_BASE_URL = "" if os.getenv("VLM_ENABLED", "1") == "0" else os.getenv("QWEN_VLM_BASE_URL", "http://127.0.0.1:8001/v1")
QWEN_VLM_MODEL = os.getenv("QWEN_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct-FP8")
QWEN_VLM_API_KEY = os.getenv("QWEN_VLM_API_KEY", "")
QWEN_VLM_TIMEOUT_SEC = float(os.getenv("QWEN_VLM_TIMEOUT_SEC", "60"))
QWEN_VLM_MAX_TOKENS = int(os.getenv("QWEN_VLM_MAX_TOKENS", "512"))
QWEN_VLM_IMAGE_MAX_PIXELS = int(os.getenv("QWEN_VLM_IMAGE_MAX_PIXELS", "147456"))
QWEN_VLM_IMAGE_JPEG_QUALITY = int(os.getenv("QWEN_VLM_IMAGE_JPEG_QUALITY", "85"))
OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "gpt-5")
OPENAI_VLM_TIMEOUT_SEC = float(os.getenv("OPENAI_VLM_TIMEOUT_SEC", "30.0"))

# 비어 있으면 개발용 로그만 남기며, 설정하면 이벤트 JSON을 POST하는 실제 1차 알림 채널이다.
# StoreOpsAI 백엔드의 /api/internal/events 로 설정하면 이 전송이 곧 DB 등록 트리거가 된다.
# 영상 경로는 상대 경로 그대로 전달하고, backend가 공유 output 볼륨에서 직접 읽어 서빙한다.
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
VLM_RESULT_WEBHOOK_URL = os.getenv("VLM_RESULT_WEBHOOK_URL", "")
ALERT_WEBHOOK_TIMEOUT_SEC = float(os.getenv("ALERT_WEBHOOK_TIMEOUT_SEC", "10.0"))

# Path 2 실시간 스트림 수신/감시 워커 설정
CONNECTION_CHECK_INTERVAL_SEC = float(os.getenv("CONNECTION_CHECK_INTERVAL_SEC", "5.0"))
STREAM_RECONNECT_DELAY_SEC = float(os.getenv("STREAM_RECONNECT_DELAY_SEC", "2.0"))

# React 개발 서버의 REST 요청 허용 목록. 운영 도메인은 배포 시 명시적으로 추가한다.
CORS_ALLOW_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",") if origin.strip()
]
