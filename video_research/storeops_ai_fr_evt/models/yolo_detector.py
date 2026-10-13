import math
from functools import lru_cache
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml

from config.config import TRACK_BUFFER_SEC, TRACKER_DIR


@lru_cache(maxsize=None)
def tracker_config_for_fps(base_config, fps):
    """track_buffer를 프레임 수가 아니라 TRACK_BUFFER_SEC(초) 기준으로 맞춘 추적 설정 파일 경로.

    기본 30프레임은 30fps에서 1초지만 3fps 영상에서는 10초가 되어,
    나갔다 들어온 다른 사람이 이전 추적 번호를 이어받는다.
    """
    cfg = yaml.safe_load(Path(check_yaml(base_config)).read_text(encoding="utf-8"))
    cfg["track_buffer"] = max(1, math.ceil(fps * TRACK_BUFFER_SEC))
    out = Path(TRACKER_DIR) / f"{Path(base_config).stem}_{cfg['track_buffer']}f.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(out)


class YOLODetector:
    def __init__(self, model_path, tracker_config="botsort.yaml", conf_threshold=0.25, person_class_id=0):
        self.model = YOLO(str(model_path))
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

    def track(self, frame, fps):
        results = self.model.track(
            frame, tracker=tracker_config_for_fps(self.tracker_config, fps), classes=[self.person_class_id],
            conf=self.conf_threshold, persist=True, verbose=False,
        )
        return results[0]
