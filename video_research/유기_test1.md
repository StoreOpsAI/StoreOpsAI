# 유기 행동 파이프라인 Test 1

- 최초 기록일: 2026-09-18
- 단순화 확정: 2026-09-19
- 기준 데이터: `data/aihub_copy`

## 1. 판정 범위

유기는 물건의 장기 잔류 여부를 추적하지 않고, 사람이 물건을 내려놓고 떠나는 **행동 자체**를 판정한다.
최종 7분류의 내부 클래스는 `유기`, 화면 표시는 `유기·무단 투기 의심`이다.

## 2. 확정 파이프라인

```text
영상
→ YOLO 사람 검출
→ BoT-SORT 사람 추적
→ 사람과 주변 맥락을 포함한 crop
→ 2~4초 행동 묶음
→ 16프레임 균등 추출
→ 공통 7분류 I3D
→ 유기 점수가 기준값 이상이면 사건 후보 생성
→ 10초 영상 저장·1차 알림·VLM 설명
```

별도 물건 검출기, YOLO-Pose, `CARRYING → PLACED → ABANDONED` 상태기계,
VideoMAE, Multi-HMR은 사용하지 않는다.

## 3. AI Hub 데이터 확인 결과

| 구분 | 영상 | XML 유기 구간 |
|---|---:|---:|
| train (`AI_data`) | 30개 | 47건 |
| validation (`AI_validation`) | 9개 | 16건 |

- 영상 규격: 1920×1080, 3 FPS, 약 180프레임
- 학습 라벨: XML의 `abandon_start`부터 `abandon_end`까지
- train/validation 분할은 원본 구성을 유지
- 생성 목록: `storeops_ai_fr_evt/output/aihub_abandonment/manifest.csv`

## 4. 스모크 테스트

구현 파일: `storeops_ai_fr_evt/pipelines/aihub_abandonment.py`

```text
AI Hub 영상
→ XML 유기 행동 구간 파싱
→ YOLO 사람 검출
→ BoT-SORT 사람 추적
→ 행동 구간과 담당 사람 추적 연결 확인
→ 결과 영상과 JSON 생성
```

첫 validation 영상 180프레임과 유기 구간 2건을 처리했다.
유기 시작 시 정답 사람 박스와 추적 박스 IoU는 `0.8193`, `0.8737`이었다.
이 결과는 행동 모델의 정확도가 아니라 데이터 파서와 사람 추적 연결을 검증한 것이다.

## 5. 다음 단계

1. 각 유기 구간에서 담당 사람의 여백 포함 crop을 만든다.
2. 2~4초 구간을 16프레임으로 균등 추출한다.
3. train 영상의 사건 밖 구간에서 정상 hard negative를 만든다.
4. 다른 다섯 이상행동 및 정상 데이터와 합쳐 공통 7분류 I3D를 학습한다.
5. train 내부 세션 분할로 만든 dev에서 유기 precision·recall과 threshold를 결정한다.
6. 학습된 I3D를 `Path1BehaviorPipeline`에 연결한다.

## 6. 재실행

```powershell
.\.venv\Scripts\python.exe storeops_ai_fr_evt\pipelines\aihub_abandonment.py `
  --smoke-video "data\aihub_copy\AI_validation\videos\C_3_11_34_BU_SMA_09-05_15-28-52_CA_RGB_DF2_M4_F4.mp4"
```

## 7. 이진 사전 검증 학습

최종 7분류 I3D 전에 데이터 흐름을 검증하기 위해 Kinetics-400 사전학습 S3D의 분류 헤드만 정상/유기로 학습했다.
이는 최종 I3D 체크포인트가 아니라 데이터와 분할 검증용 프로토타입이다.

| split | 정상 | 유기 |
|---|---:|---:|
| train | 73 | 37 |
| dev | 10 | 10 |
| test (`AI_validation`) | 16 | 16 |

- split은 원본 영상·촬영 세션 단위이며 그룹 중복은 0건이다.
- 정상 추가 추출은 train에만 적용했고 dev와 test 구성은 변경하지 않았다.
- dev에서 선택한 threshold: `0.45`
- 증강을 끈 train: accuracy `0.882`, 유기 F1 `0.827`
- dev: precision `0.909`, recall `1.000`, F1 `0.952`
- train 대비 dev 성능 하락이 없어 현재 실행에서 과적합 징후는 확인되지 않았다. 단, dev가 20클립으로 작아 최종 7분류 단계에서도 세션 단위 검증을 계속한다.
- 최종 test: precision `0.824`, recall `0.875`, F1 `0.848`
- test 혼동행렬(`[[TN, FP], [FN, TP]]`): `[[13, 3], [2, 14]]`
- 공식 test는 이 기준 결과 1회로 동결했으며, 확인된 오분류는 재학습·threshold 조정·hard negative 선정에 사용하지 않는다.

결과는 `storeops_ai_fr_evt/output/abandonment_training/`에 저장한다.

## 8. 2026-09-19 작업 종료 기록

- 유기 행동 이진 사전 검증 학습 완료
- 활성 체크포인트: `output/abandonment_training/abandonment_s3d.pt`
- 기준 체크포인트: `output/abandonment_training/abandonment_s3d_baseline.pt`
- 분할 감사: 그룹 중복 0건, 원본 영상 중복 0건
- 현재 재학습 결과에는 공식 test 평가가 포함되지 않음
- 공식 test 기준 결과는 `test_frozen.json`으로 동결
- 다음 작업은 이 분할 규칙을 유지한 공통 7분류 학습
