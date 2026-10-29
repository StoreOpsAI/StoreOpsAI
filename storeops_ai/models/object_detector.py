"""R 입력용 물건 탐지기. 연구 object_infer.detect와 같은 설정(입력 1280, 점수 하한 0.10, 프레임당 점수 순 최대 30개).
가중치가 없으면 빈 결과로 대신하지 않고 예외를 던진다(R 채널이 비면 학습 때 없던 입력이 된다).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from config.config import OBJECT_IMG_SIDE, OBJECT_INFER_CONF, OBJECT_MAX_PER_FRAME


class ObjectDetectorNotConfiguredError(RuntimeError):
    pass


class ObjectDetector:
    def __init__(self, weight_path: Path) -> None:
        if not Path(weight_path).is_file():
            raise ObjectDetectorNotConfiguredError(f"물건 탐지기 가중치가 없습니다: {weight_path}")
        from ultralytics import YOLO
        import torch
        self.model, self.half = YOLO(str(weight_path)), torch.cuda.is_available()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(frame, imgsz=OBJECT_IMG_SIDE, conf=OBJECT_INFER_CONF, half=self.half, verbose=False)[0]
        b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        order = np.argsort(-c)[:OBJECT_MAX_PER_FRAME]
        return b[order].astype(np.float32), c[order].astype(np.float32)
