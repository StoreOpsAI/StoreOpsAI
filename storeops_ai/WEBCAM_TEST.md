# 웹캠으로 학습 모델 테스트

이 프로젝트는 학습 완료된 S3D 행동분류 모델 두 개(화면 전체 A, 사람 크롭 C1)를
동적 결합해 사용합니다(가중치는 저장소에 포함하지 않으며 배치 방법은 `INTEGRATION_V1.0.md` 참고).

기본 모델:
`training/runs/ours_a_final/best.pt` + `training/runs/ours_c1_final/best.pt` +
`training/runs/ours_stack/fusion.json`

운영 화면은 정상, 쓰러짐, 쓰레기 투기, 절도 네 점수를 표시합니다. abandon은 쓰레기 투기 점수로 연결됩니다.

싸움·파손·방화는 학습하지 않았으므로 점수를 표시하지 않고 사건도 발급하지 않습니다.

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
- LITTERING
- THEFT

기본적으로 학습과 같은 4초를 모은 뒤 약 2초마다 추론합니다.
카테고리별 임계값(`fusion.json`에서 정한 값)을 넘은 비정상 클래스는 `EVENT DETECTED`로 표시합니다.

## 5. CPU가 너무 느릴 때

웹캠 화면은 별도 스레드로 계속 표시되지만, 모델 추론 자체가 CPU에서는 느릴 수 있습니다.

추론 간격을 늘리려면:

```bat
python webcam_i3d.py --interval-sec 4
```

판정 구간은 학습과 같은 4초가 기본이며 `--window-sec`로 바꿀 수 있지만, 학습 조건과 달라지므로 권하지 않습니다.

GPU를 사용하는 PC라면 프로젝트 설정의 `I3D_DEVICE=cuda`를 사용할 수 있습니다.

## 주의

이 웹캠 테스트는 **학습 때와 같은 전체 화면(full-frame) 입력**을 사용합니다.
따라서 먼저 모델 자체가 웹캠 영상에서 반응하는지 확인하는 용도입니다.

이 테스트가 정상적으로 동작하면 다음 단계에서 YOLO 사람 검출/ByteTrack과 연결하여
`사람 bbox → track ID → 행동분류 → 이벤트 후보(E001...)`까지 하나의 파이프라인으로 연결할 수 있습니다.
