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
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
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
