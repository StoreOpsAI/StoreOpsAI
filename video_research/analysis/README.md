# 전도 실시간 판정 (팀원 전달용)

이 폴더만으로 동작합니다. 모든 경로는 이 폴더 기준 상대 경로입니다.

## 구성
| 파일 | 설명 |
|---|---|
| `weights/yolo11s-pose.pt` | 사람 검출 + 관절 17개(YOLO11s-pose, 사전학습 그대로) |
| `weights/lying_mlp.pt` | 관절 56차원 → 누움 점수 MLP (66KB, 우리가 학습) |
| `label_map.json` | `normal: 0`, `fall: 1` |
| `config.json` | 3fps, 점수 임계값 0.5, 같은 자리(IoU 0.3) 1초 유지 등 (동결값) |
| `lying_core.py` | 관절 특징, MLP 로딩, 경보 규칙(과거 프레임만 사용) |
| `run_realtime.py` | 웹캠·RTSP·영상 파일 실행기 |
| `train_report.json`, `val_result.json`, `val_result_v2.json` | 학습 데이터 수·dev 5-fold 결과, 공식 validation 평가 결과(v1, v2) |
| `check_parity.py` | (개발용) 학습 코드와 배포 코드가 같은 값을 내는지 검사. 전달 시 빼도 됨 |

## 실행
```
pip install -r requirements.txt
python run_realtime.py --source 0 --show                       # 웹캠 0번, 화면 게이지
python run_realtime.py --source rtsp://user:pw@host/stream     # IP 카메라(화면 없이 경보만 stdout)
python run_realtime.py --source clip.mp4 --log-scores s.jsonl  # 영상 파일 시험, 모든 점수 기록
```
- 경보는 한 줄 JSON: `{"event":"fall","time":..,"sample":..,"box":[x1,y1,x2,y2],"score":..}`
- `--show`: 왼쪽에 **FALL 게이지**(누움 점수 0~100%, 노란 선 = 경보 임계값), 옆에 **HOLD 게이지**(같은 자리 누움이 1초 중 얼마나 이어졌는지), 사람 박스에 점수(%). q로 종료.
- 점수는 **확률이 아니라 점수**입니다(보정하지 않음).

## 동작
카메라 프레임을 3fps로 뽑아 → 포즈 모델 → 사람마다 관절 특징 → MLP 누움 점수 → 점수 ≥ 0.5인 사람이 **같은 자리에서 1초 이상** 이어지면 경보 1회. 추적 번호를 쓰지 않고 박스 위치(IoU ≥ 0.3)로 이어 붙입니다.
정의: 바닥에 누워 있는 사람. 서 있던 기록은 요구하지 않고, 쪼그려 앉기·무릎 꿇기·숙이기는 정상으로 봅니다.

## 버전
- **v1**: 증강 없음, 유지 2초 → `val_result.json` (공식 validation 1회 평가)
- **v2 (현재 `weights/lying_mlp.pt`)**: 영상 증강(저해상도·블러·밝기·JPEG, train 전용) + 유지 **1초** → `val_result_v2.json`. 같은 공식 validation을 **재사용**했으므로 독립 시험이 아니다(결과 파일 `note`에 표시).
- 결과 JSON의 `per_class` 키 `"유기"`는 공유 지표 함수의 옛 이름이며 실제로는 **누움(전도) 클래스**입니다.

| 공식 validation | v1 (2초, 증강 없음) | v2 (1초, 영상 증강) |
|---|---|---|
| 누움 precision / recall / F1 | 0.939 / 0.999 / 0.968 | 0.953 / 0.999 / 0.975 |
| 혼동(정상→누움 / 누움→정상) | 192 / 4 | 146 / 3 |
| train-val F1 차이 | 0.032 | 0.025 |
| 전도 클립 적중 | 81/81 | 81/81 |
| 매장이동 정상 오경보 | 8건 / 505.6분 (0.016/분) | 14건 / 505.6분 (0.028/분, 하루 약 40건) |

## 확인된 것 / 안 된 것
- 확인: 학습 코드와 이 폴더 코드가 같은 특징·점수·경보를 냄(`check_parity.py`), 영상 파일로 끝까지 실행됨, 게이지 화면 렌더링.
- **미확인**: 실제 웹캠 앞에서의 성능. 학습 데이터는 AI Hub의 **천장 높은 곳에서 내려다보는 1080p CCTV 영상**이라, 일반 웹캠(낮은 위치·다른 화각·해상도)에서는 정확도가 떨어질 수 있습니다. 웹캠 지연·VRAM도 아직 재지 않았습니다.
- 오경보(연출된 장면 기준)가 서비스 목표(카메라당 하루 10건 이하)보다 많을 수 있습니다. `val_result_v2.json`의 `fp` 항목을 참고하세요. 유지 시간을 1초로 줄여 경보는 빨라졌지만 오경보(0.016 → 0.028/분)가 늘었습니다.
