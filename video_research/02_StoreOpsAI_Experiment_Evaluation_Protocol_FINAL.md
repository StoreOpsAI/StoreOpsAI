# StoreOpsAI 학습 · OOF · Threshold · 실험 기록 프로토콜
## Claude Code용 — Reproducible Evaluation / Fast Experimentation / Poster Evidence

이 문서는 StoreOpsAI Vision 모델의 **학습, 평가, threshold/parameter 선택, feature cache, 실험 기록, 실패 분석, 논문/포스터 결과 생성** 규칙을 정의한다.

핵심 원칙:

> **모델/사건 구조는 사람이 정의하고, 최종 수치는 Train 데이터가 결정한다.  
> Test는 최종 검증에만 사용한다.  
> 모든 결과는 재현 가능하게 파일로 남긴다.**

---

# 0. 최종 이벤트 정의 고정

모든 학습/OOF/threshold search/Test 평가는 아래 사건 정의를 기준으로 한다.

```text
FALL
= 자세 변화 + 누운 상태 지속

ABANDONMENT
= 입장 시 외부 소지 물건 + 내려놓음 + 사람 이탈 + 물건 잔류

THEFT_SUSPICION
= 매장 물건 집음 + 소지/은닉 가능 + 미복귀 + 사람 화면 이탈

ARSON_SUSPICION
= 사람 행동 이후 근처에서 실제 Fire 발생 + temporal relation

VIOLENCE
= 두 사람 Pose 상호작용 + 타격성 motion + temporal pattern
```

실험 과정에서 False Positive를 줄이기 위해 threshold/feature/state parameter를 바꿀 수는 있지만,
이벤트 정의 자체를 Test 결과에 맞춰 바꾸지 않는다.

---

# 1. 데이터 분리

전체 데이터:

```text
FULL DATA
├─ TRAIN
└─ TEST
```

TEST는 최종 평가 전까지:
- threshold 선택
- duration 선택
- 거리 선택
- feature 선택
- fusion 선택
- rule 수치 조정

에 사용하지 않는다.

가능하면 split manifest를 최초에 고정하고 버전 관리한다.

---

# 2. 영상 leakage 방지

frame random split 금지.

최소 grouping:
- `video_id`

가능하면:
- session_id
- scene_id
- 동일 촬영 묶음
- 동일 camera/location group

중 데이터가 지원하는 더 상위 group을 사용한다.

같은 원본 영상의 거의 동일한 frame이 서로 다른 fold 또는 Train/Test에 존재하면 leakage로 간주한다.

---

# 3. Group 5-Fold OOF

TRAIN 내부에서 Group 5-Fold OOF를 수행한다.

개념:

```text
Fold 1: train 2,3,4,5 -> validate 1
Fold 2: train 1,3,4,5 -> validate 2
Fold 3: train 1,2,4,5 -> validate 3
Fold 4: train 1,2,3,5 -> validate 4
Fold 5: train 1,2,3,4 -> validate 5
```

각 sample/video는 정확히 한 번:
**자신을 학습하지 않은 모델의 prediction**을 받아야 한다.

5개 holdout prediction을 합쳐:

```text
oof_predictions
```

을 만든다.

중요:
- 각 holdout fold는 최종 Test가 아니다.
- 최종 threshold는 전체 OOF prediction을 이용해 선택할 수 있다.
- OOF metric은 tuning/validation metric이다.
- 최종 unbiased 성능은 untouched Test에서 보고한다.

---

# 3.1 Global Video Context Model Selection — S3D vs X-CLIP

S3D와 X-CLIP 중 어떤 모델을 최종 global-context backbone으로 사용할지 **Test로 결정하지 않는다.**

둘 다 동일 Group 5-Fold OOF 조건에서 평가한다.

필수 baseline:

```text
s3d_only
xclip_only
```

최소 저장:

- fold별 OOF prediction
- Event Precision
- Event Recall
- Event F1
- False Alarm / Hour
- Detection Delay
- inference latency
- peak VRAM
- model/checkpoint
- X-CLIP prompt/template 정보(사용 시)
- clip sampling 설정

선택 정책 예:

```text
Recall constraint 충족
-> False Alarm / Hour 최소
-> 동률/유사 시 F1, Delay, latency/VRAM 순으로 보조 판단
```

선택 결과는 파일로 저장한다.

```text
experiments/global_context/
  s3d_oof/
  xclip_oof/
  global_context_model_selection.json
```

`global_context_model_selection.json`에는 최소:

```text
candidate models
OOF metrics
selection objective
constraint
selected model
selected experiment_id
test_used_for_selection = false
```

를 기록한다.

최종 fusion 개발은 원칙적으로 이 **OOF-selected global model**을 primary branch로 사용한다.

다만 논문/포스터 ablation을 위해 S3D/X-CLIP 각각의 fusion 결과를 별도 실험으로 남길 수 있다.
두 경우 모두 Test 결과를 보고 선택/튜닝하지 않는다.

---

# 4. Threshold / Rule 수치 임의지정 금지

다음과 같은 임의 수치를 최종값으로 하드코딩하지 않는다.

```text
fall threshold = 0.7
duration = 1 sec
held distance = 50 px
abandon remain = 5 sec
fire threshold = 0.6
```

임시 구현 default가 꼭 필요하면:
- `TEMPORARY_DEFAULT`라고 명시
- 최종값 아님을 주석 처리
- OOF search 대상에 포함
- 최종 실험 결과 파일의 selected parameter로 교체 가능해야 함

---

# 5. 후보값 생성도 데이터 기반으로

threshold 후보조차 이유 없이 임의 grid를 만드는 것을 피한다.

가능하면 다음처럼 후보를 만든다.

## Probability / confidence

OOF prediction score 분포에서:
- unique score
- quantile
- precision-recall operating point

등을 이용해 candidate를 만든다.

## Duration

3 FPS granularity와 Train 이벤트/정상 행동 지속시간 분포를 이용한다.

예:
- annotation/event duration 분포
- hard-negative duration 분포
- 가능한 observation 단위

에서 후보를 구성한다.

## Distance

pixel distance를 사용하지 않고:
```text
distance / person_bbox_diagonal
```
또는
```text
distance / person_height
```
처럼 normalize한 Train/OOF relation 분포에서 candidate를 만든다.

## Stationary / velocity

Train object trajectory 분포를 이용한다.

즉:
**candidate space도 관측 데이터와 annotation 의미를 근거로 생성한다.**

---

# 6. 최적화 목표

StoreOpsAI의 핵심 목표는 오경보 감소이므로 기본 selection objective는:

```text
Event Recall >= target
조건을 만족하는 후보 중
False Alarm / Hour 최소화
```

Recall target은 experiment config에 명시한다.

Recall target 자체도:
- 명시적 서비스 요구
- 연구 비교 기준
- sensitivity analysis

중 하나의 근거를 가져야 한다.

가능하면 여러 Recall constraint에서 결과를 같이 저장해 trade-off를 보여준다.

---

# 7. 이벤트별 OOF tuning 대상

## Fall

- Pose probability threshold
- S3D threshold/fusion threshold
- minimum duration
- temporal window
- voting ratio
- recovery condition parameter

## Abandonment

평가 대상은 **입장 시부터 외부 소지 물건이 확인된 경우**로 한정한다.
화면 안에서 새로 집은 매장 물건은 abandonment ground truth 후보에서 제외한다.


- held distance ratio
- ownership/association threshold
- co-movement threshold
- stationary speed
- placed persistence
- separation ratio
- remain duration
- S3D abandon threshold

## Theft

절도 의심은 **보이는 상태로 소지하거나 은닉된 상태 모두 동일한 THEFT_SUSPICION 로직**으로 평가한다.
핵심은 매장 물건을 집은 뒤 미복귀 상태에서 사람이 화면을 이탈했는지 여부다.


- interaction threshold
- co-movement threshold
- return timeout
- exit persistence
- S3D theft threshold
- structured/fusion threshold

## Fire

- detector confidence
- temporal persistence
- small-fire operating point
- S3D context threshold
- fusion threshold

## Violence

- classifier probability
- person-person relation threshold
- temporal window
- persistence/duration
- S3D threshold
- fusion threshold

최종값은 OOF에서 선택한다.

---

# 8. Small Object Detector 평가/Calibration

Object detector는 event state machine 이전에 별도로 평가한다.

필수:
- Precision
- Recall
- mAP
- small-object Recall
- False Positive count

특히:
앞단 object detector가 물체를 놓치면
뒤쪽 Abandonment/Theft logic의 성능 ceiling 자체가 낮아진다.

Object confidence threshold도 Train validation/OOF prediction에 기반해 선택한다.

NMS/후처리 parameter를 조정한다면 그 역시:
- validation evidence
- detector metric
- downstream event metric

을 파일로 기록한다.

---

# 9. Feature Cache — 5개 이벤트 빠른 실험의 핵심

threshold/duration/distance를 바꿀 때마다 YOLO/S3D를 다시 돌리지 않는다.

무거운 inference와 가벼운 event search를 분리한다.

```text
VIDEO @ 3 FPS
├─ S3D
├─ X-CLIP
├─ YOLO Pose
├─ Small Object YOLO
├─ Fire/Smoke Detector
└─ Tracking
       |
       v
Feature Store
       |
       v
Repeated Event / Threshold Search
```

## 권장 feature store

예:

```text
video_id
group_id
fold_id
frame_idx
timestamp_sec
fps

person_id
person_bbox
keypoints
keypoint_confidence
pose_features
pose_event_scores

object_id
object_class
object_confidence
object_bbox
object_velocity
object_stationary_sec

wrist_object_distance_norm
person_object_distance_norm
co_movement
ownership_features

fire_score
smoke_score

s3d_scores
xclip_scores

ground_truth_event
event_start_sec
event_end_sec
```

실제 schema는 현재 데이터에 맞게 조정한다.

권장 저장:
- Parquet
- NPZ

---

# 10. OOF Feature Cache

Fold별 holdout prediction을 별도로 저장한다.

```text
cache/
  event_name/
    fold_0.parquet
    fold_1.parquet
    fold_2.parquet
    fold_3.parquet
    fold_4.parquet
    oof_all.parquet
```

metadata:
- extractor checkpoint/hash
- detector checkpoint
- git commit
- preprocessing version
- FPS
- resolution
- dataset version
- fold assignment

raw candidate generation threshold와 최종 event threshold를 구분한다.
candidate generation threshold를 너무 높게 잡아 downstream recall을 미리 죽이지 않는다.

---

# 11. 공통 Search Engine

가능하면 공통 evaluator/search interface를 구현한다.

개념:

```python
search_params(
    event_name,
    oof_features,
    search_space,
    objective="min_false_alarm_per_hour",
    constraints={"event_recall": target},
)
```

결과는 반드시:

```text
threshold_search.csv
selected_params.json
```

로 저장한다.

최종 best 값 하나만 저장하지 않는다.

---

# 12. Final Train / Test

OOF tuning 완료:

```text
OOF predictions
-> parameter search
-> selected parameters
-> FREEZE
```

그 다음:

```text
FULL TRAIN
-> final model training
-> TEST inference
-> frozen parameters 적용
```

Test에서 금지:
- threshold search
- duration 변경
- distance 변경
- feature 변경
- fusion weight 변경
- 결과를 보고 rule 수정

수정이 필요해지면:
- 해당 Test run은 기록 보존
- 새로운 실험/version으로 명시
- 원칙적으로 새로운 untouched evaluation split이 없으면 기존 Test를 반복 튜닝 대상으로 사용하지 않음

---

# 13. Event-level 평가

frame accuracy만으로 결론내리지 않는다.

필수:
- Event Precision
- Event Recall
- Event F1
- False Alarm / Hour
- False Positive Event Count
- False Negative Event Count
- Detection Delay

보조:
- frame precision/recall
- PR-AUC/AUROC
- detector mAP/Recall
- tracking 관련 지표가 필요한 경우 별도 기록

---

# 14. 모든 실험은 파일 기록

코드만 수정하고 결과를 콘솔로 끝내지 않는다.

최종 논문/포스터 발표에서 다음에 답할 수 있어야 한다.

- 어떤 데이터를 사용했는가?
- 어떤 split을 사용했는가?
- 어떤 모델을 사용했는가?
- 어떤 threshold/parameter 후보를 탐색했는가?
- 후보는 어떤 데이터 근거로 생성됐는가?
- 왜 최종 parameter가 선택됐는가?
- 기존 방식 대비 무엇이 좋아졌는가?
- 어떤 FP/FN이 있었는가?
- 어떤 수정으로 성능이 변했는가?

---

# 15. Experiment ID

모든 실험에 experiment_id를 부여한다.

예:

```text
fall_s3d_baseline_v1
fall_pose_v1
fall_temporal_v1
fall_fusion_v1

abandon_object_v1
abandon_tracking_v1
abandon_state_v1

theft_relation_v1
fire_hardnegative_v1
violence_pose_temporal_v1
```

---

# 16. Experiment별 최소 저장 항목

1. experiment_id
2. 실행 날짜/시간
3. git commit hash
4. 사용 모델
5. model checkpoint
6. dataset version
7. train/test split
8. fold 정보
9. random seed
10. FPS
11. 입력 resolution
12. 학습 hyperparameter
13. 사용 feature
14. threshold/parameter candidate 생성 방법 및 범위
15. 최종 선택 parameter
16. 선택 근거/objective/constraint
17. OOF 성능
18. Test 성능
19. FP 목록
20. FN 목록
21. Detection Delay
22. False Alarm / Hour
23. global context model 정보(S3D/X-CLIP)
24. X-CLIP prompt/template 및 backbone variant(사용 시)
25. inference latency / peak VRAM
26. 비고/known limitation

---

# 17. 권장 Experiment 디렉터리

```text
experiments/
  fall/
    fall_fusion_v1/
      config.yaml
      dataset_manifest.csv
      folds.csv

      training_metrics.json

      oof_predictions.parquet
      threshold_search.csv
      selected_params.json
      metrics_oof.json

      test_predictions.parquet
      metrics_test.json

      fp_cases.csv
      fn_cases.csv
      summary.md
```

---

# 18. Threshold Search 전체 저장

예:

```text
pose_threshold
s3d_threshold
duration_sec
window_sec
fall_ratio
precision
recall
f1
false_alarm_per_hour
detection_delay
constraint_satisfied
```

이렇게 저장해:

> "왜 이 threshold / 왜 이 duration인가?"

라는 질문에 search 결과로 답할 수 있어야 한다.

선택 이유 예:

```text
지정한 Event Recall constraint를 만족하는 OOF 후보 중
False Alarm / Hour가 최소인 조합을 선택
```

---

# 19. Ablation / 동일 Test 비교

최종적으로 동일 frozen Test set에서 비교 가능해야 한다.

## Global Context Baseline

- S3D only
- X-CLIP only

## Fall

- S3D only
- X-CLIP only
- Pose only
- Pose + Temporal
- S3D + Pose Fusion
- X-CLIP + Pose Fusion
- OOF-selected Global + Pose Fusion

## Abandonment

- S3D only
- X-CLIP only
- Object only
- Object + Tracking
- Object + Tracking + Temporal State
- S3D + Structured Fusion
- X-CLIP + Structured Fusion
- OOF-selected Global + Structured Fusion

## Theft

- S3D only
- X-CLIP only
- Object/Relation only
- Object + Tracking + Temporal
- S3D Fusion
- X-CLIP Fusion
- OOF-selected Global Fusion

## Fire

- S3D only
- X-CLIP only
- Fire detector only
- Fire + Temporal
- Fire + S3D Context/Fusion
- Fire + X-CLIP Context/Fusion
- Fire + OOF-selected Global Context

## Violence

- S3D only
- X-CLIP only
- Pose Temporal only
- S3D Fusion
- X-CLIP Fusion
- OOF-selected Global Fusion

결과는 하나의 comparison 파일로 합친다.

```text
results/ablation_summary.csv
```

columns 예:

```text
event
method
precision
recall
f1
false_alarm_per_hour
detection_delay
experiment_id
```

---

# 20. FP/FN Failure Analysis

FP/FN은 숫자만 저장하지 않는다.

각 failure case에 최소:

- case_id
- event
- predicted
- actual
- video_id
- timestamp
- 모델 score/evidence
- 추정 원인
- 수정 action
- 수정 전/후 결과
- clip/frame 경로

를 기록한다.

예:

```text
case_id: FP_001
event: fall
actual: normal
cause:
- 상품을 줍기 위해 몸을 숙임
- S3D fall score가 높음
- Pose에서는 body center가 이후 빠르게 회복

action:
- recovery temporal feature 추가

result:
- 수정 전 FP
- 수정 후 제거 여부
```

Fire 예:
```text
cause:
- 빨간 과자 포장지가 flame으로 오탐

action:
- 해당 유형을 hard negative에 추가
- detector fine-tuning

result:
- detection confidence / FP 여부 변화 기록
```

---

# 21. Failure Clip 보관

저장공간이 부족하더라도 최소:
- FP
- FN
- 대표 TP
- 대표 hard negative

clip/frame은 남긴다.

예:

```text
failure_cases/
  fall/
    fp/
    fn/
  abandonment/
    fp/
    fn/
  theft/
    fp/
    fn/
  fire/
    fp/
    fn/
  violence/
    fp/
    fn/
```

---

# 22. 성능 변화 이력

모델 수정 전/후 결과를 덮어쓰지 않는다.

예:

```text
Fall v1  S3D only
Fall v2  Pose
Fall v3  Pose + Temporal
Fall v4  Fusion
Fall v5  OOF Calibration
```

`history.csv` 예:

```text
version
experiment_id
change
precision
recall
f1
false_alarm_per_hour
detection_delay
git_commit
```

목적:
**어떤 변경이 실제 성능 개선에 기여했는지** 증명한다.

---

# 23. 개발 루프

모델 개선은 항상:

```text
문제 발견
-> 실패 사례 기록
-> 원인 가설
-> 모델/feature/state/rule 구조 수정
-> TRAIN/OOF 재평가
-> 성능 변화 기록
-> 필요 시 final model 재학습
-> frozen Test 평가
```

순서를 따른다.

Test 결과만 보고 같은 Test에 맞춰 반복 수정하지 않는다.

---

# 24. 이벤트 구현 상태

각 이벤트는 다음 상태 중 하나로 표시한다.

```text
NOT_STARTED
IMPLEMENTED
TRAINED
OOF_EVALUATED
TEST_EVALUATED
FAILED
EXPERIMENTAL
```

예:

```text
Fall: TEST_EVALUATED
Abandonment: OOF_EVALUATED
Theft: EXPERIMENTAL
Fire: TRAINED
Violence: IMPLEMENTED
```

README 또는 experiment summary에서 확인 가능하게 한다.

---

# 25. 5개 이벤트를 빠르게 실험하는 원칙

5개 이벤트 실험을 모두 빠르게 하기 위해:
**Vision inference 결과를 cache하고 event search만 반복한다.**

금지:
```text
threshold 변경
-> 영상 전체 YOLO 재실행
-> threshold 변경
-> 다시 YOLO 재실행
```

권장:
```text
영상
-> Heavy Vision inference 1회
-> feature cache
-> 수백/수천 parameter 조합을 cache에서 빠르게 평가
```

이 구조를 Fall에서 먼저 만들고,
Abandonment / Theft / Fire / Violence에 재사용한다.

---

# 26. 2주 실행 우선순위

현재 구현 상태에 따라 조정 가능하지만,
5종 모두 실험 가능한 공통 평가틀을 먼저 만든다.

권장:

```text
Day 1-2
split / GroupKFold / feature cache / common evaluator
+ S3D vs X-CLIP OOF baseline 구축

Day 3
S3D vs X-CLIP global-context OOF 비교 및 primary global model 선택

Day 4-5
Fall

Day 6-8
Abandonment

Day 9-10
Theft

Day 11
Fire

Day 12
Violence

Day 13
전체 OOF search / ablation

Day 14
Frozen Test final evaluation + Poster / failure analysis / README
```

핵심:
- 마지막 2일에 평가를 몰지 않는다.
- 각 이벤트마다 가능한 즉시 OOF 기록을 남긴다.
- 공통 cache/evaluator를 재사용한다.

---

# 27. 포스터/논문 자동 산출물

실험 결과가 쌓이면 CSV/JSON raw result를 기준으로 자동 생성 가능하게 한다.

필수 산출물:

1. S3D vs X-CLIP global-context 비교표
2. 모델별 성능 비교표
3. threshold / parameter selection 표
4. ablation table
5. FP/FN 사례 요약
6. 이벤트별 confusion matrix
7. Event Recall 비교
8. False Alarm / Hour 비교
9. Detection Delay 비교

그래프는 반드시 raw result를 먼저 저장한 뒤 생성한다.

---

# 28. 포스터에서 답할 질문

최종 발표에서 최소 다음을 답할 수 있어야 한다.

### 왜 이 threshold인가?
OOF search 결과로 답한다.

### 왜 이 duration인가?
Train/OOF duration candidate와 Event Recall / FAH trade-off로 답한다.

### 왜 이 거리인가?
normalized relation distance OOF 결과로 답한다.

### 같은 영상 frame이 train/test에 섞였는가?
split manifest / GroupKFold로 답한다.

### Test를 보고 기준을 바꿨는가?
frozen parameter와 experiment history로 답한다.

### 왜 S3D 또는 X-CLIP을 선택했는가?
Train Group 5-Fold OOF의 global-context model selection 결과로 답한다.
Test는 선택에 사용하지 않았음을 명시한다.

### 기존 방식 대비 무엇이 개선됐는가?
ablation_summary와 동일 Test metric으로 답한다.

### 오경보가 왜 발생했는가?
FP/FN failure case로 답한다.

### 어떤 수정이 효과가 있었는가?
history.csv / before-after experiment로 답한다.

---

# 29. Claude Code가 임의로 하면 안 되는 것

- 적당해 보이는 threshold를 최종값으로 지정
- 일반적으로 많이 쓴다는 이유로 duration 고정
- pixel 거리 threshold 임의 지정
- Test 결과를 보고 parameter 변경
- best score만 기록하고 search table 삭제
- 실험 결과 덮어쓰기
- console output만 남기기
- FP/FN 원인 기록 생략
- feature cache 없이 반복 Vision inference
- 동일 Test set에 반복적으로 과적합
- Test 결과로 S3D/X-CLIP 중 final global model 선택
- X-CLIP prompt를 Test 결과에 맞춰 반복 수정

---

# 30. 최종 검수 체크리스트

구현/평가가 완료되면 다음이 모두 존재하는지 확인한다.

## Data
- [ ] Train/Test manifest
- [ ] Group/Fold manifest
- [ ] dataset version
- [ ] leakage check

## Models
- [ ] S3D OOF baseline
- [ ] X-CLIP OOF baseline
- [ ] global_context_model_selection.json
- [ ] checkpoint
- [ ] training config
- [ ] training metric
- [ ] extractor version
- [ ] X-CLIP prompt/template 기록(사용 시)
- [ ] latency / peak VRAM

## OOF
- [ ] fold별 holdout prediction
- [ ] oof_all
- [ ] search candidate 전체
- [ ] selected parameter
- [ ] 선택 근거

## Test
- [ ] frozen parameter
- [ ] test prediction
- [ ] Event Precision/Recall/F1
- [ ] False Alarm / Hour
- [ ] Detection Delay

## Failure
- [ ] FP list
- [ ] FN list
- [ ] 원인
- [ ] 수정 action
- [ ] failure clip/frame

## Comparison
- [ ] S3D only
- [ ] structured only
- [ ] fusion
- [ ] event별 ablation

## Poster
- [ ] architecture
- [ ] OOF protocol
- [ ] ablation table
- [ ] FA/hour
- [ ] Detection Delay
- [ ] FP/FN 사례
- [ ] limitation

---

# 31. 최종 원칙

최종 발표에서 다음 문장을 실제 결과로 증명할 수 있어야 한다.

> 무엇을 바꿨고,  
> 왜 바꿨고,  
> 어떻게 수치를 선택했고,  
> 얼마나 개선됐고,  
> 어떤 실패가 아직 남아 있는가.

실험 결과를 덮어쓰거나 최종 점수만 남기는 방식은 금지한다.
