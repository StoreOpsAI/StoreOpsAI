"""FR-EVT 공통 설정.

현재 배포 가중치는 torchvision S3D이다. I3D로 교체하려면 I3D 구조로 재학습한
체크포인트가 필요하므로, 서비스 요구사항은 화면 전체 temporal 행동분류로 정의한다.
"""
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    """storeops_ai/.env가 있으면 읽어 환경변수로 넣는다. 이미 설정된 값은 덮어쓰지 않고, 값이 빈 줄은 건너뛴다.

    도커(compose)는 env_file로 직접 넣어 주므로 이 함수는 python으로 바로 실행할 때만 의미가 있다.
    비밀(토큰)과 컴퓨터마다 다른 주소·경로를 코드 밖 .env에 두기 위한 것이며 .env는 저장소에 올리지 않는다.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


load_env_file(BASE_DIR / ".env")

# input 폴더에 새 영상이 들어오면 사람이 API를 호출하지 않아도 자동으로 분석한다.
INPUT_DIR = Path(os.getenv("AUTO_ANALYSIS_INPUT_DIR", str(BASE_DIR / "input")))
AUTO_ANALYSIS_POLL_INTERVAL_SEC = float(os.getenv("AUTO_ANALYSIS_POLL_INTERVAL_SEC", "5.0"))
AUTO_ANALYSIS_DEFAULT_CAMERA_ID = os.getenv("AUTO_ANALYSIS_DEFAULT_CAMERA_ID", "CAM-01")


# FR-EVT-01~02
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11s-pose.pt")   # 사람 박스·추적·관절을 한 번에(2026-10-15 사용자 결정). 이전 YOLO11n은 환경변수로 되돌릴 수 있다
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", str(BASE_DIR / "config" / "botsort_storeops.yaml"))   # 데모·시연 설정(실행 폴더와 무관하게)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))   # 데모·시연 설정(BoT-SORT 논문 기본 검출 임계값 τ=0.6)
POSE_IMGSZ = int(os.getenv("POSE_IMGSZ", "1280"))   # 포즈 모델 입력 크기. 관절 헤드 P 학습 캐시(storeops_ai_fr_evt POSE_IMGSZ=1280)와 같게(2026-10-20 사용자 결정, 의사결정_정리 §19)
PERSON_CLASS_ID = 0

# FR-EVT-03~04
CLIP_MIN_SEC = float(os.getenv("CLIP_MIN_SEC", "2.0"))
CLIP_MAX_SEC = float(os.getenv("CLIP_MAX_SEC", "4.0"))
# 증거 보존용 bbox 확장 비율(확장 크롭을 쓰는 카테고리가 있을 때만 사용; 현재 학습한 카테고리에는 없다).
EXPAND_X = float(os.getenv("EXPAND_X", "0.50"))
EXPAND_Y = float(os.getenv("EXPAND_Y", "0.50"))
EXPANDED_CROP_CATEGORIES: set = set()

# 서비스 스택(2026-10-19 사용자 결정): c1p = C1(사람 크롭 S3D) + P(관절 GRU) 6클래스 동적 결합. a_c1 = 이전 A+C1 4클래스.
STACK_MODE = os.getenv("STACK_MODE", "c1p")
_C1P_DIR = BASE_DIR / "training" / "runs" / "ours_stack"
C1P_CROP_WEIGHT_PATH = os.getenv("C1P_CROP_WEIGHT_PATH", str(_C1P_DIR / "s3d_c1_6.pt"))       # 연구 finetune_s3d_c16_audit/s3d_fold-1.pt
C1P_POSE_HEAD_PATH = os.getenv("C1P_POSE_HEAD_PATH", str(_C1P_DIR / "pose_head6_rel.pt"))   # 기본 C1+(P+R)(데모 설정). C1+P는 pose_head6.pt
OBJECT_DETECTOR_PATH = os.getenv("OBJECT_DETECTOR_PATH", str(_C1P_DIR / "object_det.pt"))     # P+R 헤드일 때만 쓴다. 연구 물건 탐지기(학습·추론 설정 object_infer.py와 같게)
OBJECT_IMG_SIDE = int(os.getenv("OBJECT_IMG_SIDE", "1280"))                                  # 연구 OBJECT_FT_IMG_SIDE
OBJECT_INFER_CONF = float(os.getenv("OBJECT_INFER_CONF", "0.10"))                            # 연구 INFER_CONF(R 학습 입력과 같아야 한다)
OBJECT_MAX_PER_FRAME = int(os.getenv("OBJECT_MAX_PER_FRAME", "30"))                          # 연구 MAX_PER_FRAME
C1P_FUSION_PATH = os.getenv("C1P_FUSION_PATH", str(_C1P_DIR / "fusion_c1pr.json"))   # 기본 C1+(P+R). C1+P는 fusion_c1p.json. 연구 export_c1p.py(날짜 OOF 임계값, SYB 제외)
USE_C1P = STACK_MODE == "c1p"
if USE_C1P and not all(Path(p).is_file() for p in (C1P_CROP_WEIGHT_PATH, C1P_POSE_HEAD_PATH, C1P_FUSION_PATH)):
    raise FileNotFoundError("STACK_MODE=c1p인데 C1·P 가중치나 fusion_c1p.json이 없습니다. 파일을 두거나 STACK_MODE=a_c1로 실행하세요.")

# FR-EVT-05: 운영 점수로 노출하는 카테고리. c1p는 6클래스(파손·방화 포함), a_c1은 4클래스(정상·전도·유기·절도).
CATEGORIES = ["정상", "쓰러짐", "쓰레기 투기", "절도"] + (["파손", "방화"] if USE_C1P else [])
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
    """결합(c1p 또는 A+C1 스택)을 쓰면 OOF에서 정한 임계값(확률)을, 아니면 A 단독 기본값을 쓴다. 환경변수가 있으면 그것이 우선."""
    if USE_C1P or I3D_USE_STACK:
        import json
        spec = json.loads(Path(C1P_FUSION_PATH if USE_C1P else I3D_STACK_PATH).read_text(encoding="utf-8"))["classes"].get(category)
        if spec:
            return str(1 / (1 + math.exp(-spec["threshold_logit"])))
    return fallback

# A 단독 기본값 = OOF(재현율>=0.9, 오경보 최소)에서 고른 점수 6.23/4.70/2.15의 확률값(sigmoid). 스택 기본값은 stack.json의 OOF 임계값.
CATEGORY_THRESHOLDS = {
    "쓰러짐": _category_threshold("FALL_EVENT_THRESHOLD", _stack_default("쓰러짐", "0.998")),
    "쓰레기 투기": _category_threshold("LITTERING_EVENT_THRESHOLD", _stack_default("쓰레기 투기", "0.991")),
    "절도": _category_threshold("THEFT_EVENT_THRESHOLD", _stack_default("절도", "0.896")),
}
if USE_C1P:
    CATEGORY_THRESHOLDS["파손"] = _category_threshold("BROKEN_EVENT_THRESHOLD", _stack_default("파손", "0.999"))
    CATEGORY_THRESHOLDS["방화"] = _category_threshold("FIRE_EVENT_THRESHOLD", _stack_default("방화", "0.999"))

# FR-EVT-05: 학습된 행동분류 모델 (torchvision S3D, training/train_i3d.py 로 학습)
I3D_WEIGHT_PATH = os.getenv("I3D_WEIGHT_PATH", str(BASE_DIR / "training" / "runs" / "ours_a_final" / "best.pt"))
I3D_DEVICE = os.getenv("I3D_DEVICE", "auto")  # auto | cpu | cuda

# 학습 라벨(영문) -> 서비스 카테고리(한글). 기존 체크포인트의 미요구 클래스는 점수 응답에서 제외한다.
MODEL_LABEL_TO_CATEGORY = {
    "normal": "정상", "fall": "쓰러짐", "broken": "파손", "break": "파손",   # 6클래스 연구 체크포인트는 "break"
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
JUDGMENT_LOGIC_ENABLED = os.getenv("JUDGMENT_LOGIC_ENABLED", "1") == "1"   # 0이면 모델 점수만으로 경보(전도 3초·물건 확인·사람 없는 창 투기 제외를 모두 끈다)
STATIC_ABANDONED_CATEGORIES = set(os.getenv("STATIC_ABANDONED_CATEGORIES", "쓰레기 투기").split(","))   # 새로 놓인 물건이 있어야 경보
STATIC_REMOVED_CATEGORIES = set(os.getenv("STATIC_REMOVED_CATEGORIES", "").split(","))   # 물건이 없어져야 모델 경보. 기본 비움: 절도 모델 경보는 물건 확인 없이 낸다(사용자 결정 2026-10-21)
STATIC_SCALE = float(os.getenv("STATIC_SCALE", "0.25"))                      # 논문: 4분의 1 해상도
STATIC_PROC_FPS = float(os.getenv("STATIC_PROC_FPS", "5.0"))                 # 논문: 짧은 배경 5fps
STATIC_LONG_PERIOD_SEC = float(os.getenv("STATIC_LONG_PERIOD_SEC", "5.0"))   # 논문: 긴 배경 5초마다 갱신
STATIC_RATIO = float(os.getenv("STATIC_RATIO", "30"))                        # 논문: 짧은 배경 학습률 = 긴 배경의 30배
STATIC_MAX_EVIDENCE = int(os.getenv("STATIC_MAX_EVIDENCE", "10"))            # 5fps에서 2초(2026-10-19 사용자 결정: 경보 지연 단축, 이전 50=10초). 논문 범위 [50,500]프레임@30fps(약 1.7~17초)의 하한 쪽. 그림자·조명 오확정 증가 가능 — 시연 로그로 재검토
STATIC_DECAY =float(os.getenv("STATIC_DECAY", "1"))                         # 논문 실험값 k=1
STATIC_LONG_ALPHA = float(os.getenv("STATIC_LONG_ALPHA", "0.002"))           # 논문에 없음 — 측정으로 재검토
STATIC_MIN_AREA_FRAC = float(os.getenv("STATIC_MIN_AREA_FRAC", "0.0005"))    # 논문에 없음 — 측정으로 재검토
STATIC_HOLD_SEC = float(os.getenv("STATIC_HOLD_SEC", "30.0"))                # 논문에 없음 — 측정으로 재검토
STATIC_WRIST_CONF = float(os.getenv("STATIC_WRIST_CONF", "0.3"))            # 손목 관절을 쓸 최소 신뢰도. 논문에 없음 — 재검토
STATIC_DEPART_SEC = float(os.getenv("STATIC_DEPART_SEC", "1.0"))            # 주인 번호가 이만큼 안 보이면 떠남(사용자 요청 2026-10-20: 나가자마자 경보, 이전 2.0). 논문에 없음 — 재검토
ABANDON_UNATTEND_M = float(os.getenv("ABANDON_UNATTEND_M", "3"))   # PETS 2006 방치 거리 b(m): 물건에서 이 안에 사람이 있으면 아직 안 떠남
PERSON_HEIGHT_M = float(os.getenv("PERSON_HEIGHT_M", "1.7"))       # 픽셀→미터 근사용 평균 키(연구 쪽 config와 같다)
UNATTEND_HEIGHTS = ABANDON_UNATTEND_M / PERSON_HEIGHT_M               # 사람 키 단위 방치 거리(약 1.76)
STATIC_THEFT_OBJECT_FIRST = os.getenv("STATIC_THEFT_OBJECT_FIRST", "1") == "1"   # 1이면 물건 없어짐 + 그 근처 최근 손목만으로 절도 경보(모델 점수 무관, 사용자 결정 2026-10-20)
STATIC_ABANDON_OBJECT_FIRST = os.getenv("STATIC_ABANDON_OBJECT_FIRST", "1") == "1"   # 1이면 새 물건 + 그 물건에 닿았던 손 주인이 떠남(PETS 3m)만으로 유기 경보(모델 점수 무관, 2026-10-21)
ABANDON_CATEGORY = os.getenv("ABANDON_CATEGORY", "쓰레기 투기")   # 물건 우선 유기 경보의 클래스 이름(모델 라벨 표시명)
STATIC_PENDING_SEC = float(os.getenv("STATIC_PENDING_SEC", "30.0"))          # 모델이 먼저 이상으로 본 뒤 물건 확인을 기다리는 시간. 논문에 없음 — 측정으로 재검토
STATIC_GROW_TOL = int(os.getenv("STATIC_GROW_TOL", "10"))                    # Tian 외 2011에 수치 없음 — 측정으로 재검토
STATIC_SIMILAR_RATIO = float(os.getenv("STATIC_SIMILAR_RATIO", "1.5"))       # Tian 외 2011에 수치 없음 — 측정으로 재검토
STATIC_EDGE_RATIO = float(os.getenv("STATIC_EDGE_RATIO", "1.5"))           # 영역 키우기가 애매할 때 윤곽선 에너지 비교(Connell 외 2004, Tian 외 2011). 비율은 논문에 수치 없음 — 측정으로 재검토, 0이면 끔
STATIC_DARK_MIN = float(os.getenv("STATIC_DARK_MIN", "0.4"))                 # 빈 화면 색 모델: 밝기가 기준의 이 배 미만이면 그림자가 아니라 물건(Horprasert 외 1999 실험값 tau_alpha_lo. 논문은 정규화 값 하한이라 쓰지만 alpha에 적용 — 근거 약함)
STATIC_DETECT_SHADOWS = os.getenv("STATIC_DETECT_SHADOWS", "1") == "1"        # MOG2 그림자 판정을 켜고 그림자 픽셀은 전경에서 뺀다(Prati 외 2003 비교 근거). 0이면 이전 동작
STATIC_BG_TAU_SEC = float(os.getenv("STATIC_BG_TAU_SEC", "200"))           # 빈 화면 평균을 빛 변화에 따라가는 시간 상수(초). Tian 외 2011 alpha=0.005를 1회/초로 본 값 — 근거 약함, 0이면 끔
STATIC_HEAL_SEC = float(os.getenv("STATIC_HEAL_SEC", "60"))               # 물건 영역이 이만큼 계속 확정이면 배경으로 흡수(Tian 외 2011 healing). 경보·주인 판정(STATIC_HOLD_SEC)보다 길게. 0이면 끔
STATIC_TEXTURE_SIM = float(os.getenv("STATIC_TEXTURE_SIM", "0.7"))        # 무늬 같음 문턱(Tian 외 CVPR 2005 Ts=0.7, 논문값). 0이면 끔
STATIC_TEXTURE_WIN = int(os.getenv("STATIC_TEXTURE_WIN", "5"))            # 무늬 비교 창 크기(논문에 수치 없음 — 근거 약함)
COLOR_CLIP_LEVEL = int(os.getenv("COLOR_CLIP_LEVEL", "254"))   # 이 값 이상이면 잘린(포화) 화소(Horprasert §3). C925e MJPEG 측정: 날아간 곳이 255가 아닌 254로 나옴(2026-10-21 run29 햇빛 자국 79%가 254 이상, 255는 8%). 카메라마다 측정해 덮어쓴다
# 손-물건 접촉 탐지기(Shan 외 CVPR 2020, 100DOH 100K 가중치, 비상업 연구용). 손 상태(쥠·빈손) 근거와 숨김 후보(사용자 결정 2026-10-21). 끄면 예전과 같다
HAND_OBJECT_ENABLED = os.getenv("HAND_OBJECT_ENABLED", "0") == "1"
HAND_OBJECT_DIR = os.getenv("HAND_OBJECT_DIR", "C:/data/hand_object_detector")   # repo/(torchvision.ops로 바꾼 코드·가중치) + pydeps/(easydict). 저장소 밖
HAND_OBJECT_CONF = float(os.getenv("HAND_OBJECT_CONF", "0.5"))   # 손·물건 점수 문턱: 저장소 demo.py 기본값
# 전도 확인(사용자 지정 2026-10-19): 모델이 쓰러짐으로 본 뒤, 같은 자리에 누운 사람이 이 시간 이상 이어질 때만 경보한다. 추적 번호가 바뀌어도 위치로 잇는다
FALL_LYING_HOLD_SEC = float(os.getenv("FALL_LYING_HOLD_SEC", "3.0"))          # 사용자 지정
FALL_TORSO_DEG = float(os.getenv("FALL_TORSO_DEG", "60.0"))                  # 몸통(어깨->엉덩이)과, 발목이 보이면 다리(엉덩이->발목)도 수직에서 이 각도 이상이면 누움. 논문에 없음(근거 약함, 재검토). 쪼그려 앉기·깊이 숙이기는 걸러진다
FALL_KPT_CONF = float(os.getenv("FALL_KPT_CONF", "0.5"))                     # 어깨·엉덩이 관절 신뢰도 하한(확실한 관절만). 논문에 없음
FALL_GAP_SEC = float(os.getenv("FALL_GAP_SEC", "1.0"))                       # 관절이 잠깐 안 잡혀도 이 시간 안이면 유지로 본다. 추적기 놓친 트랙 유지와 같게: ByteTrack(Zhang 외 ECCV 2022, 본문 확인)·BoT-SORT track_buffer 30프레임 = 30fps에서 1초
FALL_SAME_SPOT = float(os.getenv("FALL_SAME_SPOT", "0.5"))                   # 박스 긴 변의 이 비율 안이면 같은 자리. 논문에 없음
# 자동 보정(버튼, 사용자 결정 2026-10-19: 어디서든 키를 누르면 자동 계산). 빈 화면 -> 걷기 순서
CALIB_KEY = os.getenv("CALIB_KEY", "c")                                      # 웹캠 창에서 이 키를 누르면 보정 시작
CALIB_EMPTY_SEC = float(os.getenv("CALIB_EMPTY_SEC", "20"))                  # 사용자 지정(20초)
TRAIN_EMPTY_LAB_MEAN = tuple(float(x) for x in os.getenv("TRAIN_EMPTY_LAB_MEAN", "55.966,0.846,5.865").split(","))   # 학습 빈 화면(224 캐시) Lab 평균, 봉인 SYB 제외 6곳 평균(측정 2026-10-20, eval_photometric.empty_lab)
TRAIN_EMPTY_LAB_STD = tuple(float(x) for x in os.getenv("TRAIN_EMPTY_LAB_STD", "19.629,8.600,12.937").split(","))     # 같은 측정의 표준편차
CALIB_LAB_MATCH = os.getenv("CALIB_LAB_MATCH", "0") == "1"   # 1이면 C1 크롭에 감마 대신 Lab 색 맞춤. 기본 끔: 웹캠(회색 방)과 학습(매장) 빈 화면의 색 퍼짐이 장면 내용 때문에 달라 채도를 과하게 올림(행동인식_분석 11-추가-7, 2026-10-20)
CALIB_EMPTY_FRAMES = int(os.getenv("CALIB_EMPTY_FRAMES", "100"))              # 빈 화면 색 모델에 쓸 프레임 수(Horprasert 외 1999의 분석이 100장)
CALIB_DETECTION_RATE = float(os.getenv("CALIB_DETECTION_RATE", "0.9999"))     # 빈 화면 픽셀이 배경으로 판정될 비율로 문턱을 고른다(Horprasert 외 1999 실험값 r)
CALIB_WALK_SEC = float(os.getenv("CALIB_WALK_SEC", "40"))                    # 사용자 지정(40초)
CALIB_MAX_SEC = float(os.getenv("CALIB_MAX_SEC", "180"))                     # 이 안에 못 끝나면 실패 처리(논문에 없음)
TRAIN_EMPTY_LUMA = float(os.getenv("TRAIN_EMPTY_LUMA", "138.5"))            # 학습 장소 7곳 빈 화면 평균 밝기의 평균(측정 2026-10-19: DYA 131.3 DYB 129.9 SMA 143.8 SMB 143.0 SYB 145.7 SMC 131.0 SYA 144.6)
TRAIN_VP = (float(os.getenv("TRAIN_VP_X", "0.0")), float(os.getenv("TRAIN_VP_Y", "1.54")))   # 학습 장소 7곳 수직 소실점(화면 중심 기준, W·H 단위)의 중앙값(연구 vertical_vp.py, 2026-10-19)
TRAIN_VP_Y_RANGE = (1.19, 1.85)                                              # 7곳의 최소·최대(측정값)
EVENT_COOLDOWN_SEC = float(os.getenv("EVENT_COOLDOWN_SEC", "30.0"))

# FR-EVT-08
EVENT_VIDEO_SEC = float(os.getenv("EVENT_VIDEO_SEC", "10.0"))
# 사건 판정 시점 기준 앞 5초 + 뒤 5초. 끝 프레임은 실시간으로 도착한 뒤 영상을 확정한다.
EVENT_VIDEO_PRE_SEC = float(os.getenv("EVENT_VIDEO_PRE_SEC", "5.0"))
EVENT_VIDEO_POST_SEC = float(os.getenv("EVENT_VIDEO_POST_SEC", "5.0"))
if abs(EVENT_VIDEO_PRE_SEC + EVENT_VIDEO_POST_SEC - EVENT_VIDEO_SEC) > 1e-9:
    raise ValueError("EVENT_VIDEO_PRE_SEC + EVENT_VIDEO_POST_SEC 는 EVENT_VIDEO_SEC 와 같아야 합니다.")

# 웹캠 캡처: 학습 영상과 같은 1920x1080(16:9). 지정하지 않으면 OpenCV가 640x480(4:3)으로 열 수 있다
CAPTURE_WIDTH = int(os.getenv("CAPTURE_WIDTH", "1920"))
CAPTURE_HEIGHT = int(os.getenv("CAPTURE_HEIGHT", "1080"))
CAPTURE_ALLOW_ASPECT = os.getenv("CAPTURE_ALLOW_ASPECT") == "1"   # 1이면 화면비가 달라도 계속(성능 보장 없음)
CAMERA_LOCK = os.getenv("CAMERA_LOCK", "1") == "1"   # 1이면 시작할 때 자동 노출·화이트밸런스·초점을 고정(정지 물건 색 모델의 전제, utils/camera_open.lock_camera)
CAMERA_LOCK_WARMUP_FRAMES = int(os.getenv("CAMERA_LOCK_WARMUP_FRAMES", "30"))   # 고정 전 최소 대기 프레임(30fps 1초). 그 뒤 밝기가 안정되면 고정(최대 300장, utils/camera_open.wait_auto_settled)
FPS_MEASURE_FRAMES = int(os.getenv("FPS_MEASURE_FRAMES", "60"))     # 이 프레임 수만큼 돈 뒤 실제 처리 속도로 시간을 다시 잰다

# FR-EVT-09~12
CAMERA_DISCONNECT_SEC = float(os.getenv("CAMERA_DISCONNECT_SEC", "60.0"))

# 저장 경로 (Path 객체여야 pipelines/path1_behavior.py 의 `CLIP_DIR / "..."` 가 동작한다)
# 클립·대표 이미지를 저장하는 곳. 탐지 서비스와 백엔드가 다른 컴퓨터일 때는 백엔드 컴퓨터의 공유 폴더를 지정한다
# (예: \\BACKEND-PC\storeops_output). 백엔드는 같은 폴더를 STOREOPS_OUTPUT_DIR로 읽는다.
# 사건 JSON(output/events)도 같은 폴더 아래에 둔다. 백엔드 컴퓨터의 탐지 서비스 컨테이너가 사건 JSON을 읽어 영상·이미지를 내려주기 때문이다.
# 환경변수를 지정하지 않으면 모두 이 컴퓨터의 output 아래에 저장한다(이전과 같음).
MEDIA_DIR = Path(os.getenv("STOREOPS_OUTPUT_DIR") or str(BASE_DIR / "output"))   # 빈 값이면 기본 폴더
CALIB_DIR = Path(os.getenv("CALIB_DIR", str(BASE_DIR / "output" / "calibration")))   # 카메라별 보정값 <camera_id>.json(이 PC 로컬). 다음 실행 때 다시 읽는다
WEBCAM_DATA_DIR = Path(os.getenv("WEBCAM_DATA_DIR", "C:/data/webcam"))   # 우리 웹캠 라벨 녹화(저장소 밖). <세션>/<라벨>_<번호>.mp4 + 같은 이름 .ts.npy(프레임 시각)
WEBCAM_LABEL_KEYS = {"0": "normal", "1": "fall", "2": "abandon", "3": "theft", "4": "break"}   # 녹화 도구 키 -> 라벨(연구 쪽 라벨 이름과 같다)
CLIP_DIR = MEDIA_DIR / "clips"
EVENT_DIR = MEDIA_DIR / "events"
IMAGE_DIR = MEDIA_DIR / "representative_images"

# FR-EVT-15
VLM_MAX_WORKERS = int(os.getenv("VLM_MAX_WORKERS", "2"))
# VLM_ENABLED=0이면 VLM을 쓰지 않는다(주소를 비움). Windows에서는 환경변수를 빈 값으로 둘 수 없어(set 변수= 는 변수를 지운다) 별도 스위치를 둔다.
# 사건이 난 순간에 VLM이 대표 이미지를 읽게 할지 여부. 기본 0 = 끔(사용자 결정 2026-10-08):
# 경보에는 사건 종류·시각·카메라와 클립·이미지만 보내고, VLM 해석은 점주가 지난 사건을 물을 때만 질문 시점에 한다.
VLM_ON_EVENT = os.getenv("VLM_ON_EVENT", "0") == "1"
# VLM 확인 뒤 경보(사용자 결정 2026-10-19, 위 10-08 결정을 바꿈): 사건 후보를 VLM이 "그 행동이 맞다"고 할 때만 백엔드로 보낸다.
# VLM_GATE_FAIL_OPEN=1이면 VLM 실패·미설정 때도 보낸다(실제 사건을 장애로 놓치지 않게). 0이면 보내지 않는다.
VLM_GATE_ENABLED = os.getenv("VLM_GATE_ENABLED", "1") == "1"
VLM_GATE_FAIL_OPEN = os.getenv("VLM_GATE_FAIL_OPEN", "1") == "1"
# 물건 우선 유기·절도(정지 물건 + 손 주인 + 떠남) 경보도 VLM 확인을 거칠지. 기본 0 = 거치지 않고 바로 보낸다(사용자 결정 2026-10-27:
# 시연에서 물건 우선 후보 3건이 모두 VLM 기각 — 약 384px 3장으로는 손에 든 물건·놓기 장면을 못 봄). 모델 경로 경보는 그대로 VLM 확인.
OBJECT_FIRST_VLM_GATE = os.getenv("OBJECT_FIRST_VLM_GATE", "0") == "1"
# Qwen 설정의 프롬프트당 이미지 제한에 맞춰 사건 확인 프레임은 3장만 저장한다.
VLM_GATE_MAX_FRAMES = int(os.getenv("VLM_GATE_MAX_FRAMES", "3"))
# 점주 대시보드 실시간 화면(utils/live_stream.py): 탐지 화면을 토큰(EVENT_INGEST_TOKEN)이 맞는 요청에만 준다. 백엔드가 점주 로그인 확인 뒤 중계한다
LIVE_STREAM_ENABLED = os.getenv("LIVE_STREAM_ENABLED", "1") == "1"
LIVE_STREAM_HOST = os.getenv("LIVE_STREAM_HOST", "0.0.0.0")   # 백엔드 컨테이너가 host.docker.internal로 부른다
LIVE_STREAM_PORT = int(os.getenv("LIVE_STREAM_PORT", "8101"))
LIVE_STREAM_WAIT_SEC = float(os.getenv("LIVE_STREAM_WAIT_SEC", "5.0"))   # 새 화면을 기다리는 최대 시간(롱 폴링). 지나면 204
# VLM에 주는 행동 판단 기준(CLAUDE.md 행동 판단 기준, 2026-09-28 사용자 확정)
BEHAVIOR_DEFINITIONS = {
    "쓰러짐": "사람이 바닥에 누워 있다(넘어졌든 쓰러졌든). 쪼그려 앉기·무릎 꿇기·허리 숙이기는 아니다.",
    "쓰레기 투기": "사람이 들고 있던 물건을 내려놓고 떠난다. 사람 없이 물건만 생기거나 사람이 서 있던 자리에 물건이 남는 것만으로는 아니다.",
    "절도": "손이 선반에 들어갔다 나오며 물건을 숨기거나, 물건이 없어지거나, 물건을 든 채 나간다.",
    "파손": "물건을 들고 바닥으로 던진다.",
    "방화": "불이나 연기가 보이고 사람이 함께 있다.",
}
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
ALERT_SHOW_SEC = float(os.getenv("ALERT_SHOW_SEC", "8"))   # 발급된 경보를 모델 창에 띄워 두는 시간(표시용, 판정과 무관)
