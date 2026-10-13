"""물건 탐지기 파인튜닝 (object_dataset으로 만든 데이터). 시험 fold는 학습·체크포인트 선택에 쓰지 않는다.

평가: python -m pipelines.eval_object_recall --detector ft --weights <best.pt> --fold <OBJECT_FT_TEST_FOLD>
"""
from __future__ import annotations

from pathlib import Path

from config.config import OBJECT_FT_BASE, OBJECT_FT_BATCH, OBJECT_FT_DIR, OBJECT_FT_EPOCHS, OBJECT_FT_IMG_SIDE, OBJECT_FT_RESUME, OBJECT_FT_RUN
from pipelines.aihub_abandonment import APP_ROOT


def main():
    from ultralytics import YOLO

    data = Path(OBJECT_FT_DIR) / "data.yaml"
    if not data.is_file():
        raise FileNotFoundError(f"데이터가 없음, object_dataset을 먼저 실행: {data}")
    last = APP_ROOT / "output" / "object_ft" / OBJECT_FT_RUN / "weights" / "last.pt"
    if OBJECT_FT_RESUME:  # 끊긴 학습을 마지막 에폭부터 이어서(설정은 저장된 것을 그대로 쓴다)
        YOLO(str(last)).train(resume=True)
        return
    model = YOLO(str(APP_ROOT / OBJECT_FT_BASE))
    model.train(data=str(data), epochs=OBJECT_FT_EPOCHS, imgsz=OBJECT_FT_IMG_SIDE, batch=OBJECT_FT_BATCH, workers=4, seed=0,
                project=str(APP_ROOT / "output" / "object_ft"), name=OBJECT_FT_RUN, exist_ok=True, patience=10)


if __name__ == "__main__":
    main()
