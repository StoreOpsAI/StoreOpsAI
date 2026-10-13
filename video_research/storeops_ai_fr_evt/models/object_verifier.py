"""새 물건 감지기가 확정한 영역이 실제 물건인지 YOLO-World(열린 어휘)로 확인한다 (유기 판단 근거 2).

배경 비교만으로는 그림자·반사·조명 변화도 '새로 생겨 멈춰 있는 영역'이 되므로,
영역을 조금 넓혀 잘라 물건 어휘(봉투·가방·컵 등)로 검출되는지 본다.
"""
from config.config import OBJECT_VERIFY_CLASSES, OBJECT_VERIFY_CONF, OBJECT_VERIFY_PAD, OBJECT_VERIFY_WEIGHTS


class ObjectVerifier:
    def __init__(self, weights=OBJECT_VERIFY_WEIGHTS, classes=OBJECT_VERIFY_CLASSES):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(str(weights))
        self.model.set_classes(list(classes))

    def verify(self, frame, box):
        """(물건 여부, 최고 점수, 어휘). 검출 박스 중심이 확정 영역 안에 있는 것만 인정한다."""
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) * OBJECT_VERIFY_PAD, (y2 - y1) * OBJECT_VERIFY_PAD
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
        crop = frame[cy1:min(h, int(y2 + dy)), cx1:min(w, int(x2 + dx))]
        if crop.size == 0:
            return False, 0.0, None
        result = self.model.predict(crop, conf=OBJECT_VERIFY_CONF, verbose=False)[0]
        best, name = 0.0, None
        for (bx1, by1, bx2, by2), score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                                    result.boxes.cls.int().tolist()):
            cx, cy = cx1 + (bx1 + bx2) / 2, cy1 + (by1 + by2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2 and score > best:
                best, name = score, result.names[cls]
        return best > 0, best, name
