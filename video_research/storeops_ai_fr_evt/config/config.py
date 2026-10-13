"""FR-EVT 공통 설정.

FR-EVT-01~04는 기존 구현을 유지하고, FR-EVT-05~16의 기능만 연결한다.
모델/학습 가중치는 이 단계에서 사용하지 않는다.
"""
import os

# FR-EVT-01~02
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "botsort.yaml")
# 사라진 사람의 추적 번호를 보관하는 시간(초). fps에 맞춰 track_buffer 프레임 수로 환산한다.
TRACK_BUFFER_SEC = float(os.getenv("TRACK_BUFFER_SEC", "1.0"))
TRACKER_DIR = "output/trackers"
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
PERSON_CLASS_ID = 0

# FR-EVT-03~04
CLIP_MIN_SEC = float(os.getenv("CLIP_MIN_SEC", "2.0"))
CLIP_MAX_SEC = float(os.getenv("CLIP_MAX_SEC", "4.0"))
# 사람 중심 + 행동 범위가 조금 더 보이는 기본값
EXPAND_X = float(os.getenv("EXPAND_X", "0.25"))
EXPAND_Y = float(os.getenv("EXPAND_Y", "0.25"))

# FR-EVT-05: 실제 모델은 아직 연결하지 않음. 7개 클래스(명세의 '여섯' 표기와 달리 실제 나열은 7개)
CATEGORIES = ["정상", "전도", "절도", "유기", "방화", "파손", "폭행"]
EVENT_THRESHOLD = float(os.getenv("EVENT_THRESHOLD", "0.60"))
CATEGORY_THRESHOLDS = {c: EVENT_THRESHOLD for c in CATEGORIES if c != "정상"}

# 유기: 없던 물건이 새로 생겨 남아 있는지 (배경 비교)
NEW_OBJECT_LONG_SEC = float(os.getenv("NEW_OBJECT_LONG_SEC", "60"))     # 장기 배경이 따라오는 시간
NEW_OBJECT_SHORT_SEC = float(os.getenv("NEW_OBJECT_SHORT_SEC", "1"))    # 단기 배경(최근 화면)이 따라오는 시간
NEW_OBJECT_DIFF = float(os.getenv("NEW_OBJECT_DIFF", "25"))             # 밝기 차 임계값(0~255)
NEW_OBJECT_PERSIST_SEC = float(os.getenv("NEW_OBJECT_PERSIST_SEC", "3"))  # 이 시간 이상 남아야 새 물건
NEW_OBJECT_MIN_AREA = int(os.getenv("NEW_OBJECT_MIN_AREA", "300"))        # 원본 해상도 기준 최소 넓이(px)
NEW_OBJECT_MAX_AREA_RATIO = float(os.getenv("NEW_OBJECT_MAX_AREA_RATIO", "0.05"))  # 화면 대비 최대 넓이
# 사람 박스를 폭·높이의 이 비율만큼 넓힌 범위와 겹치는 후보는 사람이 떠날 때까지 확정을 미룬다(팔다리·그림자 제외)
NEW_OBJECT_PERSON_MARGIN = float(os.getenv("NEW_OBJECT_PERSON_MARGIN", "0.25"))
# 테두리 윤곽선 세기(현재 / 장기 배경)가 이 값 이상이면 새로 놓인 물건, 미만이면 원래 물건이 치워진 자리
NEW_OBJECT_EDGE_RATIO = float(os.getenv("NEW_OBJECT_EDGE_RATIO", "1.0"))
# 유기 물건 확인: 확정된 새 물건 영역을 YOLO-World(열린 어휘)로 다시 보고 물건이 검출될 때만 유기로 본다(그림자·반사 제외)
OBJECT_VERIFY_WEIGHTS = os.getenv("OBJECT_VERIFY_WEIGHTS", "yolov8s-worldv2.pt")
OBJECT_VERIFY_CLASSES = os.getenv(
    "OBJECT_VERIFY_CLASSES",
    "trash bag,plastic bag,shopping bag,paper bag,bag,backpack,box,cup,bottle,can,food package,snack bag,trash,umbrella",
).split(",")
OBJECT_VERIFY_CONF = float(os.getenv("OBJECT_VERIFY_CONF", "0.05"))
OBJECT_VERIFY_PAD = float(os.getenv("OBJECT_VERIFY_PAD", "1.0"))  # 영역을 폭·높이의 이 비율만큼 넓혀 잘라 본다(주변 맥락이 있어야 검출됨)
# 유기 결합: 행동 모델이 유기라고 본 사람 근처에서, 그 구간 시작 ~ 끝 + FUSION_AFTER_SEC 안에 새 물건이 생기면 유기
FUSION_AFTER_SEC = float(os.getenv("FUSION_AFTER_SEC", "10"))
FUSION_NEAR_RATIO = float(os.getenv("FUSION_NEAR_RATIO", "0.5"))  # 사람 박스를 폭·높이의 이 비율만큼 넓혀 '근처'로 본다
# 유기 손 확인(Lin 외 TIFS 2015 역추적 응용): 물건 자리를 사람이 가리던 구간 + 그 앞 BEFORE_SEC 안에
# 손목이 물건 박스(폭·높이의 MARGIN만큼 넓힘) 안에 있었으면 사람이 놓은 것
ABANDON_HAND_BEFORE_SEC = float(os.getenv("ABANDON_HAND_BEFORE_SEC", "5"))
ABANDON_HAND_MARGIN = float(os.getenv("ABANDON_HAND_MARGIN", "1.0"))  # 유기 train 진단으로 0.5→1.0 (낙관적)
# 주인이 화면에서 사라짐: 주인 추적이 끝난 뒤 이 시간 이상 다시 나타나지 않음
OWNER_LEFT_SEC = float(os.getenv("OWNER_LEFT_SEC", "2"))
# 유기 사건 판정(PETS 2006): 소유자 a m 안 = 관리 중, b m 밖(또는 화면 밖) = 방치, 방치 t초 이어지면 경보
ABANDON_ATTEND_M = float(os.getenv("ABANDON_ATTEND_M", "2"))
ABANDON_UNATTEND_M = float(os.getenv("ABANDON_UNATTEND_M", "3"))
# PETS 2006은 30초. AI Hub 영상이 1분이라 개발 평가는 짧게 두고, 실제 매장은 환경변수로 30으로 올린다
ABANDON_GRACE_SEC = float(os.getenv("ABANDON_GRACE_SEC", "5"))
PERSON_HEIGHT_M = float(os.getenv("PERSON_HEIGHT_M", "1.7"))  # 픽셀→미터 근사용 평균 키
BEHAVIOR_WINDOW_SEC = float(os.getenv("BEHAVIOR_WINDOW_SEC", "4"))   # 행동 판단 창 길이
BEHAVIOR_STRIDE_SEC = float(os.getenv("BEHAVIOR_STRIDE_SEC", "2"))   # 창 이동 간격
# 방화: 불 검출. D-Fire 사전학습 YOLO를 방화 train 수동 불 박스(data/fire_boxes.json)로 파인튜닝한다
FIRE_BASE_WEIGHTS = os.getenv("FIRE_BASE_WEIGHTS", "weights/fire/rabahdev.pt")  # 클래스 0 smoke, 1 fire
FIRE_CLASSES = os.getenv("FIRE_CLASSES", "fire").split(",")  # 경보에 쓰는 검출 클래스 이름
FIRE_CONF = float(os.getenv("FIRE_CONF", "0.25"))             # 사전에 정한 값(dev로 고르지 않음)
FIRE_IMGSZ = int(os.getenv("FIRE_IMGSZ", "1280"))             # 작은 불을 위해 높은 입력 해상도
FIRE_SAMPLE_FPS = float(os.getenv("FIRE_SAMPLE_FPS", "1"))    # 불 검출 표본 간격
FIRE_PERSIST_SEC = float(os.getenv("FIRE_PERSIST_SEC", "2"))  # 이 시간 이상 연속 검출돼야 불
FIRE_NEAR_RATIO = float(os.getenv("FIRE_NEAR_RATIO", "0.5"))  # 사람+불: 사람 박스를 이 비율만큼 넓힌 범위 안에 불
# 관절 추정(파손 던지기 손목 속도, 전도 몸통 각도)
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11s-pose.pt")
POSE_IMGSZ = int(os.getenv("POSE_IMGSZ", "1280"))  # 1080p 클립에서 작은 사람도 잡도록 큰 입력
POSE_MIN_CONF = float(os.getenv("POSE_MIN_CONF", "0.5"))  # 이 신뢰도 미만 관절은 쓰지 않는다
# 파손 = 집기 → 던지기 → 떨어짐. 던지기 손목 속도(몸통 길이/표본)는 라벨 관절 분석에서 반품 1%·파손 80% 지점
BROKEN_THROW_SPEED = float(os.getenv("BROKEN_THROW_SPEED", "0.97"))
BROKEN_AFTER_SEC = float(os.getenv("BROKEN_AFTER_SEC", "5"))   # 던진 뒤 이 안에 바닥 새 물건
BROKEN_PICK_SEC = float(os.getenv("BROKEN_PICK_SEC", "10"))    # 던지기 전후 이 안에 선반 물건이 치워진 자리
# 전도 = 누워 있는 사람 분류기(동결 CLIP 이미지 인코더 + 선형), Lafuente-Arroyo 외 ESWA 2022 2단계 방식 응용
LYING_BACKBONE = os.getenv("LYING_BACKBONE", "openai/clip-vit-base-patch32")
AIHUB_ORGANIZED_ROOT = os.getenv("AIHUB_ORGANIZED_ROOT", "C:/data/aihub/organized")  # 정리된 AI Hub 전도·매장이동 (행동/split/videos|labels)
LYING_FIRST_PASS_CLIPS = int(os.getenv("LYING_FIRST_PASS_CLIPS", "300"))  # 첫 학습에 쓸 전도 클립 수(날짜 그룹 단위로 채움)
LYING_TRANSITION_FRAMES = int(os.getenv("LYING_TRANSITION_FRAMES", "15"))  # 전도 시작 앞뒤(3fps 15프레임=5초)는 전환 구간이라 학습·평가에서 뺀다(사전 값)
LYING_POSITIVE_IOU = float(os.getenv("LYING_POSITIVE_IOU", "0.3"))  # 라벨 전도 박스와 이만큼 겹치면 누운 사람
LYING_MLP_HIDDEN = tuple(int(v) for v in os.getenv("LYING_MLP_HIDDEN", "128,64").split(","))  # 관절 특징 MLP 은닉 크기(사전 값, 튜닝 안 함)
LYING_MLP_EPOCHS = int(os.getenv("LYING_MLP_EPOCHS", "60"))
ANALYSIS_DIR = os.getenv("ANALYSIS_DIR", "../analysis")  # 팀원에게 보낼 배포 폴더(APP_ROOT 기준 상대 경로)
LYING_AUG_COPIES = int(os.getenv("LYING_AUG_COPIES", "3"))  # train 샘플마다 만들 증강 복사본 수(사전 값). dev·val·test에는 적용하지 않는다
LYING_AUG_ROT_DEG = float(os.getenv("LYING_AUG_ROT_DEG", "10"))  # 카메라 기울기 흉내: 관절·박스 회전 ±도
LYING_AUG_JITTER = float(os.getenv("LYING_AUG_JITTER", "0.02"))  # 관절 흔들림(사람 큰 변 대비 표준편차)
LYING_AUG_DROPOUT = float(os.getenv("LYING_AUG_DROPOUT", "0.05"))  # 관절을 안 보이게 만드는 비율(가림 흉내)
LYING_AUG_BOX_SCALE = float(os.getenv("LYING_AUG_BOX_SCALE", "0.1"))  # 검출 박스 가로·세로 크기 흔들림 ±비율
ABANDON_FIRST_PASS_CLIPS = int(os.getenv("ABANDON_FIRST_PASS_CLIPS", "321"))  # 유기 첫 학습 클립 수(전도 first_pass와 같은 수, 세션 그룹 단위로 채움)
LYING_HOLD_SEC = float(os.getenv("LYING_HOLD_SEC", "1"))  # 같은 자리 누움이 이 시간 이상 이어지면 전도 경보. TEMPORARY_DEFAULT. 사용자 확정(2026-10-06): 1,2,3,4,5초 중 OOF 탐색(search_lying_params)으로 고른다

# 행동 백본 비교 (얼린 사전학습 영상 모델, HuggingFace 모델 ID)
VIDEOMAE_MODEL = os.getenv("VIDEOMAE_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics")
XCLIP_MODEL = os.getenv("XCLIP_MODEL", "microsoft/xclip-base-patch32")

# FR-EVT-08
EVENT_VIDEO_SEC = float(os.getenv("EVENT_VIDEO_SEC", "10.0"))

# FR-EVT-09~12
CAMERA_DISCONNECT_SEC = float(os.getenv("CAMERA_DISCONNECT_SEC", "60.0"))

# 저장 경로
CLIP_DIR = "output/clips"
EVENT_DIR = "output/events"
IMAGE_DIR = "output/representative_images"

# FR-EVT-15
VLM_MAX_WORKERS = int(os.getenv("VLM_MAX_WORKERS", "2"))

# 물건 탐지기 재현율 측정(eval_object_recall): 라벨 물건 박스와의 IoU 기준, 탐지 최소 점수, 추론 해상도
OBJECT_RECALL_IOU = float(os.getenv("OBJECT_RECALL_IOU", "0.3"))
OBJECT_RECALL_CONF = float(os.getenv("OBJECT_RECALL_CONF", "0.05"))
OBJECT_RECALL_IMGSZ = int(os.getenv("OBJECT_RECALL_IMGSZ", "1280"))

# 물건 탐지기 파인튜닝(object_dataset / train_object): 데이터 위치, 시험·검증 fold, 프레임 간격, 이미지 긴 변, 학습 설정
OBJECT_FT_DIR = os.getenv("OBJECT_FT_DIR", "C:/data/object_ft")
OBJECT_FT_TEST_FOLD = int(os.getenv("OBJECT_FT_TEST_FOLD", "0"))   # 안 본 세션 시험 fold(학습·체크포인트 선택에 안 씀)
OBJECT_FT_VAL_FOLD = int(os.getenv("OBJECT_FT_VAL_FOLD", "1"))     # 체크포인트 선택용 fold
OBJECT_FT_FRAME_STRIDE = int(os.getenv("OBJECT_FT_FRAME_STRIDE", "2"))
OBJECT_FT_IMG_SIDE = int(os.getenv("OBJECT_FT_IMG_SIDE", "1280"))
OBJECT_FT_EPOCHS = int(os.getenv("OBJECT_FT_EPOCHS", "30"))
OBJECT_FT_BASE = os.getenv("OBJECT_FT_BASE", "yolo11s.pt")

# 물건 탐지기 오경보 측정(eval_object_false_alarms): 점수 기준들, 같은 자리 박스로 볼 IoU, 정상 영상 수
OBJECT_FA_CONFS = tuple(float(x) for x in os.getenv("OBJECT_FA_CONFS", "0.25,0.5,0.7").split(","))
OBJECT_FA_SAME_IOU = float(os.getenv("OBJECT_FA_SAME_IOU", "0.5"))
OBJECT_FA_VIDEOS = int(os.getenv("OBJECT_FA_VIDEOS", "24"))

# 유기 시간 창 특징 추출(abandon_windows / abandon_features): 창 길이·간격(프레임, 3fps), 창에 사람이 있어야 하는 비율,
# 사건 전후 여유(프레임), 사람 크롭 여유(박스 폭·높이의 비율, 한 변당), 특징 저장 위치
ABANDON_WIN_FRAMES = int(os.getenv("ABANDON_WIN_FRAMES", "24"))
ABANDON_WIN_STRIDE = int(os.getenv("ABANDON_WIN_STRIDE", "12"))
ABANDON_WIN_MIN_PRESENT = float(os.getenv("ABANDON_WIN_MIN_PRESENT", "0.5"))
ABANDON_EVENT_MARGIN = int(os.getenv("ABANDON_EVENT_MARGIN", "12"))
ABANDON_CROP_PAD = float(os.getenv("ABANDON_CROP_PAD", "0.5"))
ABANDON_FEATURE_DIR = os.getenv("ABANDON_FEATURE_DIR", "C:/data/features/abandon_v1")

# 물건 탐지기 매장 단위 시험(배경 일반화 확인): 지정하면 이 사이트의 영상 전부가 시험(학습·선택에 안 씀). 실행 이름도 바꿔 덮어쓰지 않게 한다
OBJECT_FT_TEST_SITE = os.getenv("OBJECT_FT_TEST_SITE", "")
OBJECT_FT_RUN = os.getenv("OBJECT_FT_RUN", "run")

# 행동 공통 특징 추출(behavior_features / extract_behavior): 행동별 학습 영상 수(세션 단위로 채움), fold 수, 정리된 영상 위치
BEHAVIOR_TRAIN_CLIPS = int(os.getenv("BEHAVIOR_TRAIN_CLIPS", "100000"))
BEHAVIOR_FOLDS = int(os.getenv("BEHAVIOR_FOLDS", "5"))
BEHAVIOR_FEATURE_ROOT = os.getenv("BEHAVIOR_FEATURE_ROOT", "C:/data/features")

# 시간 창 MLP 학습(train_windows): 은닉 크기, 드롭아웃, 가중치 감쇠, 에폭, 학습률, 시드들 (사전 값, 튜닝하지 않음)
WIN_MLP_HIDDEN = int(os.getenv("WIN_MLP_HIDDEN", "128"))
WIN_MLP_DROPOUT = float(os.getenv("WIN_MLP_DROPOUT", "0.3"))
WIN_MLP_WEIGHT_DECAY = float(os.getenv("WIN_MLP_WEIGHT_DECAY", "1e-3"))
WIN_MLP_EPOCHS = int(os.getenv("WIN_MLP_EPOCHS", "40"))
WIN_MLP_LR = float(os.getenv("WIN_MLP_LR", "1e-3"))
WIN_MLP_SEEDS = tuple(int(x) for x in os.getenv("WIN_MLP_SEEDS", "0,1,2").split(","))
OBJECT_FT_RESUME = os.getenv("OBJECT_FT_RESUME", "") == "1"  # 1이면 마지막 체크포인트에서 이어서 학습

# 사람·물건 연결 특징(object_relation / abandon_relations): 탐지 점수 하한, 물건 박스 이어 붙이는 IoU, 쥠 거리(사람 박스 높이 비율),
# 놓음 거리, 정지로 볼 이동량(사람 박스 높이 비율), 정지 최소 프레임, 등장 직후로 볼 프레임 수 (사전 값, 튜닝하지 않음)
REL_DET_CONF = float(os.getenv("REL_DET_CONF", "0.25"))
REL_LINK_IOU = float(os.getenv("REL_LINK_IOU", "0.3"))
REL_HOLD_DIST = float(os.getenv("REL_HOLD_DIST", "0.25"))
REL_RELEASE_DIST = float(os.getenv("REL_RELEASE_DIST", "0.5"))
REL_STATIC_MOVE = float(os.getenv("REL_STATIC_MOVE", "0.1"))
REL_STATIC_FRAMES = int(os.getenv("REL_STATIC_FRAMES", "3"))
REL_ENTRY_FRAMES = int(os.getenv("REL_ENTRY_FRAMES", "6"))
ABANDON_REL_DIR = os.getenv("ABANDON_REL_DIR", "C:/data/features/abandon_v1_rel")

# 사람-물건 관계 시계열(abandon_cache / relation_sequence): 탐지 저장 하한(모델이 점수를 입력으로 받으므로 낮게), 캐시 위치, 등장 직후 프레임 수
CACHE_DET_CONF = float(os.getenv("CACHE_DET_CONF", "0.1"))
ABANDON_CACHE_DIR = os.getenv("ABANDON_CACHE_DIR", "C:/data/features/abandon_v1_cache")
SEQ_ENTRY_FRAMES = int(os.getenv("SEQ_ENTRY_FRAMES", "6"))

# 전도 운영점 선택(select_lying_operating_point): 목표 오경보율(카메라당 하루 건수, 제품 요구), 하루 분 수, 후보 기준·유지 시간(초)
LYING_FA_BUDGETS_PER_DAY = tuple(float(x) for x in os.getenv("LYING_FA_BUDGETS_PER_DAY", "10,2").split(","))
MINUTES_PER_DAY = float(os.getenv("MINUTES_PER_DAY", "1440"))
LYING_GRID_THRESHOLDS = tuple(float(x) for x in os.getenv("LYING_GRID_THRESHOLDS", "0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.995").split(","))
LYING_GRID_HOLDS = tuple(float(x) for x in os.getenv("LYING_GRID_HOLDS", "1,2,3,4,5").split(","))
LYING_LINK_IOU = float(os.getenv("LYING_LINK_IOU", "0.3"))  # TEMPORARY_DEFAULT: 같은 자리로 이어 붙이는 박스 IoU. OOF 탐색 대상(search_lying_params)이며 선택값으로 교체한다
LYING_RECALL_TARGETS = tuple(float(x) for x in os.getenv("LYING_RECALL_TARGETS", "0.9,0.95,0.98").split(","))  # 재현율 제약 민감도 분석용(여러 값을 같이 저장해 trade-off를 보인다). 서비스 요구가 정해지면 교체
PERCEPTION_FPS = float(os.getenv("PERCEPTION_FPS", "3"))  # 스펙 01 §0-3: 기본 운영·학습 시간 해상도. 영상 fps가 더 높으면 이 간격으로 표본을 뽑는다
OBJECT_FT_BATCH = int(os.getenv("OBJECT_FT_BATCH", "8"))  # 다른 GPU 작업과 함께 돌릴 때는 줄인다
