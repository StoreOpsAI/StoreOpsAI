# Project Instructions

## 영상 학습 데이터 분할 규칙

데이터 누수와 과적합을 막기 위해 다음 규칙을 반드시 지킨다.

1. 하나의 원본 영상에서 만든 모든 2~4초 창은 반드시 같은 split에 둔다. 창 단위 무작위 분할은 금지한다.
2. 같은 촬영 회차와 같은 장면의 다중 카메라 영상은 하나의 그룹으로 묶어 같은 split에 둔다.
3. augmentation은 train 데이터에만 적용한다.
4. 정상 구간은 이상행동 구간 전후에 여유 구간을 두고 추출해 전환 프레임이 섞이지 않게 한다.
5. 공식 validation 데이터는 최종 시험용으로 보존하고, threshold 조정에 반복 사용하지 않는다.
6. threshold 조정과 개발 검증은 train 내부를 촬영 세션 단위로 나눠 수행한다.
7. test 오분류 샘플은 모델·전처리·threshold 수정이나 hard negative 선정에 재사용하지 않는다. test 평가는 설정을 동결한 뒤 한 번만 수행한다.

분할 후에는 원본 영상, 촬영 세션, 다중 카메라 그룹이 train과 validation/test 사이에 겹치지 않는지 자동 검사한다. 모델 학습 결과에는 클래스별 precision, recall, F1, 혼동행렬과 train-validation 성능 차이를 기록한다.
