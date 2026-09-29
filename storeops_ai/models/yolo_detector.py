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
