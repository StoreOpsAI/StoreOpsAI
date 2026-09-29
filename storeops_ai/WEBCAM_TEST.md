# 웹캠으로 학습 모델 테스트

이 프로젝트에는 학습 완료된 S3D 행동분류 모델이 포함되어 있습니다.

기본 모델:
`training/runs/mc_stage2/best.pt`

학습 라벨은 다음 6개입니다.
- normal → 정상
- fall → 전도
- broken → 파손
- fire → 방화
- abandon → 유기
- theft → 절도

현재 체크포인트에는 `fight(폭행)` 학습 클래스가 없으므로 폭행은 이 모델로 감지되지 않습니다.

## 1. 프로젝트 루트

Windows CMD/PowerShell에서:

```bat
cd C:\storeops_ai
```

또는 실제 압축을 푼 프로젝트 경로로 이동합니다.

## 2. 가상환경 활성화

```bat
venv\Scripts\activate
```

이미 활성화되어 있으면 생략합니다.

## 3. 실행

```bat
python webcam_i3d.py
```

웹캠 번호가 0이 아니면:

```bat
python webcam_i3d.py --camera 1
```

## 4. 화면에서 확인할 것

웹캠을 켜고 사람이 행동하면 최근 영상 구간을 S3D 모델에 넣어 다음 점수를 표시합니다.

- NORMAL
- FALL
- BROKEN
- FIRE
- ABANDON
- THEFT
- FIGHT (항상 0.0)

기본적으로 8초를 모은 뒤 약 2초마다 추론합니다.
점수가 0.60 이상인 비정상 클래스는 `EVENT DETECTED`로 표시합니다.

## 5. CPU가 너무 느릴 때

웹캠 화면은 별도 스레드로 계속 표시되지만, 모델 추론 자체가 CPU에서는 느릴 수 있습니다.

추론 간격을 늘리려면:

```bat
python webcam_i3d.py --interval-sec 4
```

판정 구간을 10초로 바꾸려면:

```bat
python webcam_i3d.py --window-sec 10
```

GPU를 사용하는 PC라면 프로젝트 설정의 `I3D_DEVICE=cuda`를 사용할 수 있습니다.

## 주의

이 웹캠 테스트는 **학습 때와 같은 전체 화면(full-frame) 입력**을 사용합니다.
따라서 먼저 모델 자체가 웹캠 영상에서 반응하는지 확인하는 용도입니다.

이 테스트가 정상적으로 동작하면 다음 단계에서 YOLO 사람 검출/ByteTrack과 연결하여
`사람 bbox → track ID → 행동분류 → 이벤트 후보(E001...)`까지 하나의 파이프라인으로 연결할 수 있습니다.
