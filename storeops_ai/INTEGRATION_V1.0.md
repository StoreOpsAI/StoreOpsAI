# StoreOps AI 영상 모델 v1.0 연동 안내 (VLM 미사용)

기존 단일 S3D 분류기 대신 **화면 전체(A) + 사람 크롭(C1) S3D 두 개를 동적 결합**한 분류기를 `storeops_ai/`에 연결합니다.
백엔드·프런트엔드는 수정하지 않습니다(`category`를 아래 이름으로 보내면 기존 `CATEGORY_EVENT_TYPES`와 프런트 표시 이름이 그대로 동작).

## 서비스 카테고리 (v1.0)

| 서비스 점수 | 학습 라벨 | 백엔드 event_type |
| --- | --- | --- |
| 정상 | normal | - |
| 쓰러짐 | fall | fall |
| 쓰레기 투기 | abandon | littering |
| 절도 | theft | theft |

싸움·파손·방화는 v1.0에서 제외합니다(학습하지 않음). 6클래스(파손·방화 포함) 학습은 v1.1에서 다룹니다.

## 변경 파일

| 파일 | 내용 |
| --- | --- |
| `config/config.py` | 카테고리 4개, C1 가중치·결합 파일 경로, 임계값을 `fusion.json`(OOF로 정한 값)에서 읽기, `VLM_ENABLED` 스위치, `CROP_LARGEST_PERSON` |
| `models/stack_head.py` (신규) | A+C1 동적 결합 `z = (a\|a\| + b\|b\|) / (\|a\| + \|b\|)` (학습 파라미터 없음), 창 안 사람 박스 합집합 크롭 |
| `models/i3d_classifier.py` | C1 로드, `predict_pair()` |
| `pipelines/path1_behavior.py` | 사람이 없는 창도 분류, C1 크롭 입력, `risk_level`·`score_logit` 기록 |
| `schemas/event_schema.py` | `risk_level`, `score_logit` 필드(선택) |
| `requirements.txt` | `uvicorn[standard]` (WebSocket 사용) |
| `tests/test_i3d_classifier.py`, `webcam_i3d.py` | 4카테고리 기준, 웹캠 점검 도구 |
| `../docker-compose.novlm.yml` (신규) | VLM 없이 `storeops-ai`를 띄우는 덮어쓰기 파일 |

## 가중치 (저장소에 포함하지 않음)

다음 파일을 `storeops_ai/training/runs/` 아래에 같은 이름으로 둡니다(도커 빌드 컨텍스트에 포함되도록 `COPY . .` 전에 배치).

| 경로 | 설명 |
| --- | --- |
| `ours_a_final/best.pt`, `label_map.json` | A(화면 전체) S3D, 4클래스 |
| `ours_c1_final/best.pt`, `label_map.json` | C1(사람 크롭) S3D, 4클래스 |
| `ours_stack/fusion.json` | 동적 결합과 카테고리별 임계값(OOF 5-fold, 재현율 목표 0.9, 전도 0.95) |

없으면 `I3D_USE_STACK`이 꺼지고 A 단독(`I3D_WEIGHT_PATH`)으로 동작합니다. 가중치가 전혀 없으면 `I3DNotConfiguredError`를 냅니다(임의 점수를 만들지 않음).

## 실행 (VLM 없이)

```powershell
docker compose -f docker-compose.yml -f docker-compose.novlm.yml up -d --build
```

- VLM을 끄는 환경변수는 `VLM_ENABLED=0`입니다(Windows에서는 빈 환경변수가 변수를 지우므로 별도 스위치를 둠).
- 이벤트는 `ALERT_WEBHOOK_URL`로 사건이 났을 때만 한 번 전송됩니다(주기 전송 아님). 같은 카테고리는 `EVENT_COOLDOWN_SEC`(30초) 동안 재발급하지 않습니다.

## 동작 방식과 주의

- 3fps로 쌓은 화면을 **2초마다 최근 4초**로 분류합니다. 사람 게이트는 두지 않습니다(쓰러져 추적이 끊긴 사람, 사람이 떠난 직후의 유기를 놓치지 않기 위해).
- 임계값은 확률이 아니라 점수 기준으로 정했고 `fusion.json`의 로그 오즈를 시그모이드로 환산해 씁니다. 현재 기본: 쓰러짐 4.91, 쓰레기 투기 4.51, 절도 3.28(로그 오즈).
- **실시간 웹캠 모드(`pipelines/path1_webcam.py`)는 아직 A 단독, 사람 게이트, 3회 연속 확인 방식 그대로입니다.** 이번 변경은 파일 분석 경로(`path1_behavior.py`)에 적용됩니다. 실시간 모드에 같은 결합을 적용하는 작업은 후속입니다(`--confirmations 1`로 연속 확인은 낮출 수 있음).
- 평가 한계: 공식 validation을 여러 번 보았으므로 블라인드가 아니며, 새 환경(강의실) 성능은 별도 측정이 필요합니다. 근거와 수치는 연구 저장소의 `행동인식_분석.md`, `의사결정_정리.md` 참고.
- 검증: 단위 테스트 13개 통과(`PYTHONPATH=.;.. python -m unittest discover -s tests`), 컴포즈 덮어쓰기 파일은 `docker compose config`로 문법 확인. 도커 이미지 빌드와 실제 컨테이너 기동은 확인하지 못했습니다. 도커 이미지는 CPU용 torch라서 CPU 추론 시간(이 PC 기준 모델 2개 합쳐 2초 창당 약 0.6초)을 배포 환경에서 다시 확인해야 합니다.
