## 네이밍 규칙 (Python / FastAPI)

### 함수
- API 핸들러: `[verb]_[entity]` snake_case, 조회 목록은 `list_`, 단건은 `get_`
  - 예: `get_event`, `list_events`, `get_event_clip`, `analyze_video`
- 헬퍼 함수: `[verb]_[noun]` snake_case, 모듈 내부 전용은 `_` 접두사
  - 예: `save_frames`, `send_first_alert`, `_read_crop_clip`
- 클래스: PascalCase, 약어는 대문자 유지
  - 예: `EventManager`, `YOLODetector`, `I3DClassifier`
- 예외: PascalCase + `Error` (예: `I3DNotConfiguredError`)
- pydantic 요청/응답 모델: `[Action]Request` / `[Entity]` (예: `AnalyzeRequest`, `EventCandidate`)
- 상수·설정값: UPPER_SNAKE_CASE, `config/config.py`에 둔다 (예: `EVENT_THRESHOLD`)

### 파일
- 모듈: snake_case + `.py` (예: `event_manager.py`, `yolo_detector.py`)
- API 라우터: `api/routes_[path].py` (예: `routes_path1.py`)
- 파이프라인: `pipelines/[대상]_[역할].py` (예: `path1_behavior.py`, `train_abandonment.py`)
- 테스트: `tests/test_[모듈명].py` (예: `tests/test_event_manager.py`)

## 금지 사항
- 타입 힌트에 `Any` 남용 금지. 외부 JSON(VLM 응답 등) 경계에서만 허용한다.
- 300줄 넘는 모듈, 80줄 넘는 함수 새로 작성 금지. 넘으면 나눈다.
- 임계값·경로·URL·카메라 ID 하드코딩 금지. `config/config.py`에 두고 환경변수로 덮어쓸 수 있게 한다.
- 모델 가중치가 없을 때 임의 점수 생성 금지. 명시적 예외를 던진다.
