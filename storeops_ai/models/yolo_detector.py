from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path, tracker_config="bytetrack.yaml", conf_threshold=0.25, person_class_id=0):
        self.model = YOLO(str(model_path))
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

    def track(self, frame):
        results = self.model.track(
            frame, tracker=self.tracker_config, classes=[self.person_class_id],
            conf=self.conf_threshold, persist=True, verbose=False,
        )
        return results[0]

    @staticmethod
    def keypoints_by_track(result) -> dict:
        """추적 번호 -> (관절 좌표 17x2, 관절 신뢰도 17). 포즈 모델이 아니거나 번호가 없으면 빈 딕셔너리."""
        if result.keypoints is None or result.keypoints.xy is None or result.boxes is None or result.boxes.id is None:
            return {}
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else [None] * len(xy)
        return {int(tid): (xy[i], conf[i]) for i, tid in enumerate(result.boxes.id.cpu().numpy())}
