#!/bin/bash
# 다중 클래스(전도/폭행/파손/방화/절도) 학습 파이프라인
#
# 컨테이너 안에서 실행:
#   bash scripts/run_pipeline.sh
#
# 전제: docker run 시 아래처럼 마운트되어 있을 것
#   -v ~/aihub_all:/workspace/raw/train        <- 학습용 mp4+xml
#   -v ~/aihub_all_val:/workspace/raw/val      <- 검증용 mp4+xml
#   -v ~/scripts:/workspace/scripts
#   -v ~/runs:/workspace/runs
#   -v ~/datasets:/workspace/datasets

set -e  # 에러 나면 즉시 중단

FRAMES=24
ARCH=s3d
RAW_TRAIN=raw/train
RAW_VAL=raw/val
DS_TRAIN=datasets/train
DS_VAL=datasets/val

# 폭행(fight)은 출처가 NIA2019로 달라 배경 shortcut 위험이 커서 베이스라인에서 제외.
# 나중에 AI-Hub 폭행 영상을 구하면 여기서 빼면 된다.
EXCLUDE=fight
# 정상 클립이 다른 클래스(영상당 1개, 약 30개)와 비슷해지도록 비율 조정
NORMAL_RATIO=0.2

echo "=============================================="
echo "0단계: 이벤트 길이 분포 확인 (선택)"
echo "=============================================="
if [ -f scripts/analyze_event_durations.py ]; then
  python scripts/analyze_event_durations.py --labels $RAW_TRAIN --videos $RAW_TRAIN \
    --no-video-fps || echo "(건너뜀)"
else
  echo "analyze_event_durations.py 없음 - 건너뜀 (참고용이라 학습엔 영향 없음)"
fi

echo
echo "=============================================="
echo "1단계: 프레임 추출 (train / val 동일 옵션)"
echo "=============================================="
python scripts/extract_multiclass.py \
  --videos $RAW_TRAIN --labels $RAW_TRAIN --out $DS_TRAIN \
  --frames $FRAMES --extract_normals --normals_per_video 1 \
  --normal_video_ratio $NORMAL_RATIO --exclude_classes $EXCLUDE

python scripts/extract_multiclass.py \
  --videos $RAW_VAL --labels $RAW_VAL --out $DS_VAL \
  --frames $FRAMES --extract_normals --normals_per_video 1 \
  --normal_video_ratio $NORMAL_RATIO --exclude_classes $EXCLUDE

echo
echo "=============================================="
echo "2단계: 배경 shortcut 위험 진단 (중요)"
echo "=============================================="
python scripts/check_shortcut_risk.py --data $DS_TRAIN

echo
echo ">>> 위 진단 결과를 확인하세요."
echo ">>> [위험] 항목이 많으면 학습해도 실전에서 작동하지 않을 수 있습니다."
read -p ">>> 계속하려면 Enter, 중단하려면 Ctrl+C: "

echo
echo "=============================================="
echo "3단계: 백본 동결 학습 (head만)"
echo "=============================================="
python scripts/train_i3d.py \
  --data $DS_TRAIN --val_data $DS_VAL \
  --arch $ARCH --epochs 30 --lr 1e-3 --batch_size 8 \
  --freeze_backbone --out runs/mc_stage1

echo
echo "=============================================="
echo "4단계: 전체 미세조정"
echo "=============================================="
python scripts/train_i3d.py \
  --data $DS_TRAIN --val_data $DS_VAL \
  --arch $ARCH --epochs 20 --lr 1e-4 --batch_size 8 \
  --resume runs/mc_stage1/best.pt --out runs/mc_stage2

echo
echo "=============================================="
echo "완료"
echo "=============================================="
echo "체크포인트: runs/mc_stage2/best.pt"
echo
echo "다음 단계:"
echo "  1) 클래스별 정확도를 확인하세요 (전체 정확도보다 중요)"
echo "  2) 절도/방화는 낮게 나오는 것이 정상입니다"
echo "  3) 학습에 쓰지 않은 원본 영상으로 실제 추론 테스트를 하세요"
