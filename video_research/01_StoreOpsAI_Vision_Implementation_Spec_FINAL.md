# StoreOpsAI Vision 이상탐지 구현 명세
## Claude Code용 — Architecture / Modeling / Event Logic

이 문서는 StoreOpsAI의 Vision 이상탐지 부분을 구현하기 위한 최상위 명세다.
Claude Code는 구현 전에 **현재 workspace/branch 전체 코드를 먼저 읽고**, 현재 코드가 이 문서보다 구체적인 구현 사실을 가지고 있다면 그 사실을 확인한 뒤 이 명세와 정합되게 수정한다.

---

# 0. 절대 원칙

1. **현재 로컬 workspace/branch가 source of truth다.**
   - GitHub main과 로컬이 다르면 로컬을 우선한다.
   - S3D, YOLO Pose, tracker 등이 이미 구현되어 있다고 추측하지 말고 실제 코드를 확인한다.

2. 이번 작업 범위는 **Vision 모델링 + 이벤트 판정**까지다.
   - LLM Agent orchestration은 범위 밖이다.
   - POS 실제 연동, 재고 Agent, 보고서 Agent 구현은 범위 밖이다.
   - Vision 결과를 상위 시스템이 소비할 수 있는 구조화 event/evidence 형태로 출력하는 데까지만 구현한다.
   - 현재 API / EventCandidate 또는 동등한 event schema / backend 연동이 이미 존재한다면 **불필요하게 깨지 않는다.**
   - 새 Vision evidence가 필요하면 기존 event contract를 최대한 유지하면서 확장 필드 또는 adapter로 연결한다.

3. 기본 운영/학습 시간 해상도는 **3 FPS**다.
   - 시간 조건을 frame count로 하드코딩하지 않는다.
   - 모든 temporal rule은 seconds 기반으로 표현한다.
   - 내부적으로 `duration_sec = observations / perception_fps`처럼 계산한다.

4. **구조는 사람이 정의하고, 수치는 데이터가 결정한다.**
   - 사건 상태전이 구조는 설계자가 정의할 수 있다.
   - confidence, threshold, duration, distance, ratio, persistence, NMS/후처리 기준 등 최종 수치는 임의로 정하지 않는다.
   - 최종 수치 선택 규칙은 `02_StoreOpsAI_Experiment_Evaluation_Protocol.md`를 따른다.

5. YOLO Pose / S3D / Small-object detector / Fire detector를 서로 대체 관계로 보지 않는다.
   - 서로 다른 evidence를 제공하는 병렬 Vision branch로 설계한다.

6. 동일 영상의 유사 frame이 학습/검증/Test에 섞이는 leakage를 허용하지 않는다.

---

# 최종 이벤트 정의 — 이 정의를 기준으로 구현

아래 5개 이벤트 정의를 최종 기준으로 사용한다.
이벤트 의미를 임의로 확장하거나 다른 정의로 바꾸지 않는다.

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

중요:
- 위 정의는 사건 구조 자체다.
- threshold, distance, duration, confidence, ratio 같은 수치는 이 정의에 포함되지 않는다.
- 최종 수치는 Train 내부 Group 5-Fold OOF에서 결정한다.
- Test 결과를 보고 위 구조 또는 수치를 다시 조정하지 않는다.

---

# 1. 전체 목표 아키텍처

```text
                           CCTV / VIDEO
                                |
                        Common 3 FPS Sampling
                                |
          +---------------------+----------------------+
          |                     |                      |
          v                     v                      v
  Global Video Context     YOLO Pose            Small Object YOLO
   S3D or X-CLIP          Person + Pose          Small Objects
          |                     |                      |
          |                     +----------+-----------+
          |                                |
          |                       Tracking / Association
          |                                |
          |                 +--------------+--------------+
          |                 |                             |
          |                 v                             v
          |          Pose / Relation Features       Object Features
          |                 |                             |
          +-----------------+--------------+--------------+
                                            |
                                            v
                                    Feature / Evidence Layer
                                            |
                         +------------------+------------------+
                         |                                     |
                         v                                     v
                   Fire/Smoke Branch                    Temporal Engine
                         |                                     |
                         +------------------+------------------+
                                            |
                                            v
                                        Event Fusion
                                            |
              +-------------+-------------+-------------+-------------+-------------+
              |             |             |             |             |
              v             v             v             v             v
             FALL      ABANDONMENT       THEFT       FIRE/ARSON    VIOLENCE
                                            |
                                            v
                                   Structured Event Output
```

---

# 2. Global Video Context Branch — S3D vs X-CLIP

전체 장면의 최근 수 초가 어떤 행동처럼 보이는지 판단하는 **global-context branch**는
처음부터 S3D 하나로 고정하지 않는다.

후보:

- S3D
- X-CLIP

두 모델을 동일한 데이터 split / 동일한 Group 5-Fold OOF / 동일한 event-level 평가 기준에서 비교한 뒤,
**Train OOF 결과로 primary global-context backbone을 선택한다.**

절대 Test 성능을 보고 S3D와 X-CLIP 중 하나를 고르지 않는다.

---

## 2.1 S3D 역할

S3D는 **전체 장면의 최근 수 초가 어떤 행동처럼 보이는지** 나타내는 global-context branch다.

예:
- normal
- fall
- abandon
- theft
- fire
- violence/fight

중요:
- 현재 workspace에 S3D가 있으면 재사용/정리한다.
- 없으면 baseline/global-context branch로 구현할 수 있게 설계한다.
- **S3D score 하나만으로 모든 최종 사건을 확정하지 않는다.**
- 포스터/논문에서는 S3D-only baseline과 structured/fusion 방식을 비교할 수 있어야 한다.

S3D 기본 입력 정책은 현재 코드/학습 설정을 먼저 확인한다.
현재 baseline이 `3 FPS / 4초 window / 2초 overlap / full-scene` 구조라면 그 설정을 **baseline 재현 설정으로 보존하고 기록**한다.
다만 workspace가 다르면 4초/2초를 사실로 가정하지 말고 실제 구현값을 사용한다.
window와 overlap도 실험 설정 및 experiment metadata에 반드시 기록한다.

---

## 2.2 X-CLIP 역할

X-CLIP은 CLIP 계열의 사전학습 표현을 비디오 인식에 활용하는 global video-context 후보로 사용한다.

목적:
- 적은 라벨 환경에서 강한 pretrained representation 활용
- 장면 전체의 semantic action context 제공
- S3D 대비 일반화 성능 비교
- Pose/Object/Tracking이 제공하는 구조적 evidence와 결합

X-CLIP은 아래를 대체하지 않는다.

- YOLO Pose
- Small Object YOLO
- Fire/Smoke detector
- Person/Object tracking
- explicit state transition

즉 X-CLIP은 유기/절도에서 "누가 어떤 물체를 들고 있었고 언제 놓았는지"를 직접 보장하는 모델이 아니라
**장면 수준 semantic evidence를 제공하는 global branch**다.

### X-CLIP 학습/평가 원칙

- 현재 workspace에 구현이 없다면 독립 classifier interface로 추가한다.
- fine-tuning / linear probing / 필요 시 zero-shot 또는 few-shot baseline은 experiment_id를 분리한다.
- text prompt를 사용할 경우 prompt template을 실험 설정에 반드시 저장한다.
- prompt를 Test 결과에 맞춰 반복 수정하지 않는다.
- input frame sampling, clip length, resolution, backbone variant를 experiment metadata에 저장한다.
- S3D와 비교할 때 동일 Train/Test split과 동일 OOF protocol을 사용한다.

### 하드웨어 측정

X-CLIP은 S3D보다 무거울 수 있으므로 정확도만으로 선택하지 않는다.
최소 다음을 기록한다.

- peak VRAM
- clip inference latency
- throughput
- model parameter count 또는 모델 크기
- 3 FPS 운영에서의 실제 처리 가능성

단, 최종 global-context backbone 선택의 1차 기준은 연구 objective이며,
하드웨어 지표는 deployment feasibility의 보조 기준으로 기록한다.

---

## 2.3 S3D vs X-CLIP 선택 규칙

두 global branch를 Train 내부 Group 5-Fold OOF에서 비교한다.

최소 비교:

```text
S3D only
X-CLIP only
```

평가:

- Event Precision
- Event Recall
- Event F1
- False Alarm / Hour
- Detection Delay
- inference latency
- peak VRAM
```

기본 선택 원칙:

```text
Event Recall constraint를 만족하는 후보 중
False Alarm / Hour가 더 낮은 모델 우선
```

동률/유사 성능이면:
- Event F1
- Detection Delay
- latency / VRAM
- 구현 안정성

을 보조 근거로 사용한다.

선택 결과는 `global_context_model_selection.json`에 기록한다.

예:

```json
{
  "candidates": ["s3d", "xclip"],
  "selected": "xclip",
  "selection_source": "train_oof",
  "objective": "min_false_alarm_per_hour_under_recall_constraint",
  "test_used_for_selection": false
}
```

위 값은 schema 예시이며 실제 selected 모델은 OOF 결과가 결정한다.

최종 Test에서는:
- S3D baseline
- X-CLIP baseline
- structured-only
- selected-global + structured fusion

을 동일 frozen Test set에서 비교할 수 있게 한다.

---

# 3. YOLO Pose 역할 및 학습

YOLO Pose는:
- person bbox
- keypoints
- keypoint confidence
- person confidence

를 제공한다.

YOLO Pose가 person bbox를 안정적으로 제공한다면 별도의 person detector를 중복 실행하지 않는다.

Tracker와 연결해:
- person_id
- bbox history
- pose history
- trajectory
- screen enter/exit state

를 유지한다.

## 공통 Pose 모델 원칙

행동별로 Pose 모델을 계속 덮어쓰지 않는다.

금지:
```text
Fall 데이터로 fine-tune
-> Fall 원본 삭제
-> Violence 데이터만 이어서 fine-tune
-> Theft 데이터만 이어서 fine-tune
```

권장:
```text
normal
+ fall
+ violence
+ theft
+ abandonment
+ fire/arson 관련 사람 pose
-> 통합 Store Pose Fine-tuning
-> best_store_pose.pt
```

YOLO Pose의 목적은 행동 label 자체를 직접 기억하는 것이 아니라,
**편의점/무인매장 환경의 다양한 사람 자세에서 안정적으로 bbox/keypoint를 추출하는 공통 perception model**을 만드는 것이다.

---

# 4. 저장공간이 부족한 경우의 Pose 데이터 처리

원본 영상을 모두 영구 보관하지 않아도 된다.

행동별로:

```text
원본 영상
-> Pose 학습용 frame + annotation 추출
-> keypoint/metadata/feature 추출
-> 추출 무결성 검증
-> 필요한 실패분석 clip 보존
-> 원본 영상 삭제 가능
```

단, 원본 삭제 전에 반드시 다음이 복원 가능해야 한다.

- video_id
- timestamp/frame index
- label
- event start/end
- split
- fold
- Pose 학습용 image/annotation
- keypoint sequence
- feature
- 사용한 extractor checkpoint/version

---

# 5. Small Object YOLO 역할 및 학습

목적:
- 손에 든 작은 물체
- 바닥/선반에 놓인 작은 물체
- 유기/절도 상태전이에 필요한 물체
- 방화 관련 소형 도구가 annotation 가능한 경우 해당 객체

일반 detector가 전체 CCTV 화면에서 작은 물체를 놓치는 문제가 있기 때문에 small-object 중심 fine-tuning을 별도 branch로 둔다.

학습 데이터는 "손에 들고 있는 물체"만 포함하면 안 된다.

반드시 가능한 범위에서:
- 손에 든 물체
- 내려놓는 중
- 바닥에 놓인 물체
- 선반 위 물체
- 일부 가려진 물체
- 정상 구매 중 물체
- 오탐을 유발하는 hard negative

를 포함한다.

## ROI inference

YOLO Pose의 wrist/person bbox를 활용한 ROI inference를 선택적으로 지원할 수 있다.

```text
full frame
-> person / wrist ROI
-> crop
-> Small Object YOLO
```

단, ROI는 최적화 옵션이며 첫 구현을 불필요하게 복잡하게 만들지 않는다.

---

# 6. Fire / Smoke Detector

Fire/Smoke detector는:
- 작은 불꽃
- 가능한 경우 smoke

를 탐지한다.

다음 hard negative를 적극 포함한다.

- 조명
- 반사광
- 빨강/주황 포장
- 과자 봉지
- 전광판
- 유리 반사
- 화면 밝기/노출 변화
- 불꽃처럼 보이는 작은 고휘도 영역

Fire detection과 Arson 판단은 분리한다.

```text
Fire detected != Arson confirmed
```

---

# 7. Tracking / Association

사람과 물체의 ID 및 시간적 연속성을 유지한다.

현재 tracker가 있으면 재사용 여부를 먼저 판단한다.
ByteTrack / BoT-SORT 등 실제 workspace 구현을 확인하고 선택한다.

## PersonTrack

- person_id
- bbox history
- pose history
- trajectory
- velocity
- enter/exit state

## ObjectTrack

- object_id
- class
- bbox history
- trajectory
- velocity
- stationary duration

## Relation feature

- normalized wrist-object distance
- normalized person-object distance
- object inside/near person bbox
- co-movement
- ownership association
- object original zone
- object return 여부
- person/object exit 상태

픽셀 절대거리 기준은 사용하지 않는다.
거리 normalization과 수치 결정은 실험 프로토콜을 따른다.

---

# 8. 공통 Feature / Evidence 구조

가능하면 공통 schema를 만든다.

예:

```text
VisionEvidence
- video_id
- timestamp_sec
- frame_idx
- person_ids
- object_ids

- s3d_scores

- pose:
  - keypoints
  - confidences
  - pose score/features

- object:
  - class
  - bbox
  - confidence
  - velocity
  - stationary duration

- relation:
  - wrist_object_distance_norm
  - person_object_distance_norm
  - co_movement
  - ownership

- fire:
  - fire_score
  - smoke_score

- temporal:
  - state history
  - duration

- ground_truth / metadata
```

이 결과는 반복 실험용 feature cache로 저장할 수 있어야 한다.

---

# 9. FALL / 전도

## 최종 사건 정의

```text
FALL
= 자세 변화 + 누운 상태 지속
```

Pose/S3D/X-CLIP/temporal feature는 위 정의를 검증하기 위한 evidence로 사용한다.


## 목적

정상 행동:
- 상품 줍기
- 몸 숙이기
- 쪼그리기
- 앉기
- 일시적으로 바닥 가까이 내려가기

를 전도로 오인하는 것을 줄이면서 실제 전도 Recall을 유지한다.

## 입력

- YOLO Pose
- person tracking history
- S3D fall score
- 3 FPS temporal sequence

## Pose feature 후보

- normalized keypoints
- torso angle
- shoulder/hip geometry
- bbox aspect ratio
- body center height
- hip/shoulder height
- vertical displacement
- joint/keypoint velocity
- recovery behavior

## 구조

```text
YOLO Pose
-> Pose feature
-> Fall classifier (MLP / 필요 시 TCN/LSTM)
-> fall probability
-> temporal evidence
-> S3D context
-> fusion
-> FALL
```

단일 frame 또는 개발자가 임의로 정한 "연속 N frame"만으로 확정하지 않는다.

최종:
- classifier threshold
- minimum duration
- temporal window
- voting ratio
- S3D threshold/fusion 기준

은 OOF에서 선택한다.

---

# 10. ABANDONMENT / 유기

## 최종 사건 정의

```text
ABANDONMENT
= 입장 시 외부 소지 물건 + 내려놓음 + 사람 이탈 + 물건 잔류
```

화면 안에서 새로 집은 매장 물건은 유기 대상이 아니다.
그 경우는 정상 구매 또는 절도 쪽 로직으로 보낸다.


유기는 단일 frame classification이 아니라 **state transition** 문제로 처리한다.

```text
Person P1 + Object O1
-> ASSOCIATED / HELD
-> PLACED
-> OBJECT_STATIONARY
-> PERSON_SEPARATING
-> PERSON_EXITED
-> OBJECT_REMAINS
-> ABANDONMENT
```

## 중요한 요구사항

CCTV 화면에 처음 등장했을 때 이미 들고 있는 물체도 association한다.

동시에:
화면 내부에서 새로 집은 물체도 association 가능해야 한다.

## HELD evidence 후보

- wrist-object normalized distance
- object near/inside person bbox
- person/object co-movement
- association persistence

## PLACED evidence 후보

이전 association/held 상태 이후:
- person-object separation 증가
- object velocity 감소
- object 위치 안정화

## ABANDONED evidence

- object stationary
- person-object separation
- person exit
- object remains

최종 거리/시간/속도 기준은 OOF에서 결정한다.

## Hard negatives

- 잠깐 내려놓고 다시 집기
- 가방을 내려놓고 다시 들기
- 상품을 선반에 반환
- 정상 구매 중 일시적 물체 분리

---

# 11. THEFT / 절도

## 최종 사건 정의

```text
THEFT_SUSPICION
= 매장 물건 집음 + 소지/은닉 가능 + 미복귀 + 사람 화면 이탈
```

소지 상태에서 물체는 계속 보일 수도 있고, 몸/가방 쪽에서 가려져 더 이상 직접 검출되지 않을 수도 있다.
둘은 같은 절도 로직 안에서 처리한다.


Vision만으로 범죄를 확정하지 않는다.

내부적으로는 `THEFT_SUSPICION` 개념을 사용한다.
기존 서비스 호환 때문에 출력 label이 "절도"인 경우에도 evidence metadata에 suspicion임을 남긴다.

## 상태 흐름

```text
store/shelf object
-> person approaches
-> hand-object interaction
-> object picked
-> object removed from original location
-> person-object co-movement
-> object not returned
-> exit zone / screen exit
-> THEFT_SUSPICION
```

`held_object == theft` 금지.

정상 구매 행동은 가장 중요한 hard negative다.

- 정상 상품 집기
- 상품 비교
- 상품 반환
- 매장 내부에서 들고 이동
- 장바구니/가방에 정상적으로 담기
- 계산대 방향 이동

## Future extension

POS 실제 연동은 이번 범위 밖이다.

향후:
```text
payment_verified: Optional[bool]
```

같은 interface 확장 가능성만 고려한다.

---

# 12. FIRE / ARSON

## 최종 사건 정의

```text
ARSON_SUSPICION
= 사람 행동 이후 근처에서 실제 Fire 발생 + temporal relation
```

Fire detector는 실제 불꽃/연기를 확인하는 perception evidence이고,
사람 행동과 Fire 발생의 시간적/공간적 관계를 함께 본다.


## Fire event

```text
Fire/Smoke detector
-> temporal evidence
-> FIRE EVENT
```

단일 frame 오탐을 줄이기 위한 persistence 등의 최종 수치는 데이터에서 선택한다.

## Arson suspicion

가능한 evidence:
- fire onset
- S3D context
- person proximity
- person-object interaction
- interaction 이후 fire onset
- temporal order

```text
fire detected == arson
```

으로 구현하지 않는다.

Fire event와 Arson suspicion을 내부적으로 분리 가능한 구조로 둔다.

---

# 13. VIOLENCE / 폭행

## 최종 사건 정의

```text
VIOLENCE
= 두 사람 Pose 상호작용 + 타격성 motion + temporal pattern
```

두 사람의 관절 좌표, 상대 거리, 관절 속도/가속도, 상대방 방향 motion, 반복적 타격 패턴을 evidence로 사용한다.


폭행은 Pose temporal interaction 중심으로 설계한다.

## 입력

- Person A Pose
- Person B Pose
- tracking history
- S3D violence score

## Feature 후보

- normalized person-person distance
- wrist velocity
- ankle velocity
- joint acceleration
- body-center relative velocity
- 상대방 방향 movement
- 반복적 strike-like motion
- interaction persistence

## Classifier

초기에는 데이터 규모와 구현 안정성을 보고:
- engineered temporal feature + MLP
- TCN
- LSTM

중 가장 단순한 방식부터 시작한다.

ST-GCN은 확장 후보이며 초기 필수 사항은 아니다.

## Hard negatives

- 빠르게 지나가기
- 장난
- 악수
- 물건 전달
- 가까운 거리 대화
- 빠른 정상 손동작

---

# 14. NORMAL / Hard Negative Pool

정상 데이터를 버리지 않는다.

공통 normal/hard negative pool:

- standing
- walking
- normal shopping
- reaching
- product picking
- product returning
- bending
- temporary object placing
- fast but normal hand movement

연결:

```text
bending / crouching
-> Fall hard negative

normal product picking
-> Theft hard negative

temporary placing
-> Abandonment hard negative

fast normal interaction
-> Violence hard negative

light / reflection / snack package
-> Fire hard negative
```

---

# 15. Fusion

초기에는 이유 없는 단순 평균 금지.

금지:
```python
final_score = (s3d + pose + object) / 3
```

비교 가능한 모드:

```text
s3d_only
xclip_only
structured_only
fusion_s3d
fusion_xclip
fusion_selected
```

`fusion_selected`는 Train OOF에서 선택된 global-context backbone만 사용한다.
Test 성능을 보고 S3D/X-CLIP을 선택해서는 안 된다.

structured_only:
- Pose/Object/Tracking/Temporal/Relation 기반

fusion_s3d:
- S3D global context + structured evidence

fusion_xclip:
- X-CLIP global context + structured evidence

fusion_selected:
- Train OOF에서 선택된 global-context model + structured evidence

초기:
- evidence gate / state transition

데이터가 충분하면:
- Logistic Regression
- small MLP

등 learned fusion으로 확장 가능하다.

Learned fusion 역시 Train OOF prediction만으로 학습/보정한다.

---

# 16. 공통 출력

Vision 모델링의 최종 출력은 상위 시스템이 소비할 수 있는 구조화 결과다.

예:

```json
{
  "event": "abandonment",
  "confidence": 0.0,
  "video_id": "...",
  "person_id": 0,
  "object_id": 0,
  "start_time_sec": 0.0,
  "detected_time_sec": 0.0,
  "evidence": {
    "s3d": {},
    "pose": {},
    "object": {},
    "relation": {},
    "temporal": {}
  }
}
```

위 숫자는 schema 예시일 뿐이며 threshold 예시가 아니다.

---

# 17. 코드 구조

현재 package layout을 먼저 확인하고 억지로 재구성하지 않는다.
다만 책임은 다음처럼 분리한다.

```text
models/
  pose_detector
  s3d_classifier
  xclip_classifier
  small_object_detector
  fire_detector

tracking/
  person_tracker
  object_tracker

features/
  pose_features
  object_features
  relation_features

events/
  fall_detector
  abandonment_detector
  theft_detector
  fire_event_detector
  violence_detector

fusion/
  event_fusion

schemas/
  vision_evidence

training/
  pose/
  object/
  fire/
  fall/
  violence/

evaluation/
  event_metrics
  threshold_search
  oof_evaluation
  feature_cache
```

---

# 18. 구현 순서

Claude Code는 먼저 현재 repository를 분석한다.

첫 분석 결과에 반드시 포함:

1. 현재 Vision pipeline 실제 호출 흐름
2. YOLO Pose 구현 여부/위치
3. S3D 구현 여부/위치
4. tracker 구현
5. 3 FPS sampling 위치
6. 현재 데이터 split 구조
7. 현재 수동 threshold/heuristic 목록
8. 재사용 가능한 모듈
9. 수정할 모듈
10. 새로 만들 모듈
11. 기존 API/Event schema에 미치는 영향
12. 공통 feature cache를 어디에 추가할지

그 다음 구현한다.

권장 순서:

```text
STEP 1  공통 VisionEvidence / cache schema
STEP 2  S3D + X-CLIP + Pose + Object + Fire inference interface
STEP 3  S3D vs X-CLIP Train-OOF baseline benchmark 및 global branch 선택
STEP 4  Fall
STEP 5  Abandonment
STEP 6  Theft
STEP 7  Fire/Arson
STEP 8  Violence
STEP 9  Fusion
STEP 10 통합 OOF / event-level evaluation
STEP 11 ablation / poster result export
```

각 이벤트를 모두 독립 프로젝트처럼 새로 만들지 말고,
공통 inference/cache/evaluation framework를 재사용한다.

---

# 19. 구현 금지사항

- 현재 workspace를 확인하지 않고 기존 구현을 추측
- 행동별 YOLO Pose 모델을 순차 덮어쓰기
- frame random split
- Test 결과를 보고 threshold/rule 수정
- Test 결과를 보고 S3D/X-CLIP 중 global backbone 선택
- pixel 고정 거리로 person-object relation 판단
- held_object == theft
- object placed == abandonment
- fire detected == arson
- 이유 없는 score 평균
- 30 FPS 기준 temporal frame 수 하드코딩
- threshold 변경마다 전체 Vision inference 재실행
- LLM/VLM을 주요 사건 판정기로 대체
- 구현되지 않은 기능을 완료된 것처럼 문서화

---

# 20. 완료 조건

Vision 구현 자체의 완료 조건은 단순히 코드가 실행되는 것이 아니다.

최종적으로 최소 다음을 지원해야 한다.

- 3 FPS 기반 inference
- S3D/X-CLIP/Pose/Object/Fire evidence
- person/object temporal tracking
- 5개 이벤트별 structured detector
- feature cache
- `s3d_only / xclip_only / structured_only / fusion_s3d / fusion_xclip / fusion_selected` 비교
- Group 5-Fold OOF 기반 parameter selection
- untouched Test evaluation
- FP/FN failure analysis
- 포스터용 결과 export

학습/평가/threshold search 방법과 기록 규칙은 반드시 두 번째 문서를 따른다.
