"""없던 물건이 새로 생겨 남아 있는지 찾는 배경 비교 감지기 (유기 판단 근거 1).

장기 배경과는 다르고 최근 화면(단기 배경)과는 같은 영역 = 새로 생겨서 멈춰 있는 것.
사람 박스 안은 제외하고, NEW_OBJECT_PERSIST_SEC 이상 유지되고 근처에 사람이 없을 때(놓고 떠난 뒤) 확정한다.
확정한 영역은 테두리 윤곽선으로 '새로 놓인 물건(added)'과 '원래 있던 물건이 치워진 자리(removed)'를 구분한다.
"""
import cv2
import numpy as np

from config.config import (
    NEW_OBJECT_DIFF, NEW_OBJECT_EDGE_RATIO, NEW_OBJECT_LONG_SEC, NEW_OBJECT_MAX_AREA_RATIO, NEW_OBJECT_MIN_AREA,
    NEW_OBJECT_PERSIST_SEC, NEW_OBJECT_PERSON_MARGIN, NEW_OBJECT_SHORT_SEC,
)

SCALE = 0.5  # 속도를 위해 절반 해상도에서 계산하고 박스는 원본 좌표로 돌려준다


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class NewObjectDetector:
    def __init__(self, fps):
        self.fps = fps
        self.long_alpha = 1.0 / max(1.0, fps * NEW_OBJECT_LONG_SEC)
        self.short_alpha = min(1.0, 1.0 / max(1.0, fps * NEW_OBJECT_SHORT_SEC))
        self.persist_frames = max(1, round(fps * NEW_OBJECT_PERSIST_SEC))
        self.long_bg = None
        self.short_bg = None
        self.candidates = []  # {"box", "first", "last"}
        self.frame_no = -1

    def update(self, frame, person_boxes):
        """프레임 하나를 넣고, 이번 프레임에 새로 확정된 물건 박스 목록을 돌려준다."""
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        return self.update_gray(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0), person_boxes)

    def update_gray(self, blurred_gray, person_boxes):
        """절반 해상도 + 블러를 거친 회색 프레임(uint8)을 받는다. 파라미터 탐색에서 영상을 다시 디코딩하지 않으려고 분리했다."""
        self.frame_no += 1
        gray = blurred_gray.astype(np.float32)
        if self.long_bg is None:
            self.long_bg, self.short_bg = gray.copy(), gray.copy()
            return []

        person = np.zeros(gray.shape, np.uint8)
        for x1, y1, x2, y2 in person_boxes:
            w, h = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            person[max(0, int((y1 - h) * SCALE)):int((y2 + h) * SCALE),
                   max(0, int((x1 - w) * SCALE)):int((x2 + w) * SCALE)] = 1

        cv2.accumulateWeighted(gray, self.short_bg, self.short_alpha)
        cv2.accumulateWeighted(gray, self.long_bg, self.long_alpha, mask=1 - person)
        static_new = ((np.abs(gray - self.long_bg) > NEW_OBJECT_DIFF)
                      & (np.abs(gray - self.short_bg) < NEW_OBJECT_DIFF) & (person == 0)).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        static_new = cv2.morphologyEx(cv2.morphologyEx(static_new, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        min_area = NEW_OBJECT_MIN_AREA * SCALE * SCALE
        max_area = NEW_OBJECT_MAX_AREA_RATIO * gray.size
        count, _, stats, _ = cv2.connectedComponentsWithStats(static_new)
        boxes = [(x / SCALE, y / SCALE, (x + w) / SCALE, (y + h) / SCALE)
                 for x, y, w, h, area in stats[1:count] if min_area <= area <= max_area]
        self._gray, self._mask = gray, static_new
        return self._confirm(boxes, person_boxes)

    def _confirm(self, boxes, person_boxes):
        near_person = [(x1 - (x2 - x1) * NEW_OBJECT_PERSON_MARGIN, y1 - (y2 - y1) * NEW_OBJECT_PERSON_MARGIN,
                        x2 + (x2 - x1) * NEW_OBJECT_PERSON_MARGIN, y2 + (y2 - y1) * NEW_OBJECT_PERSON_MARGIN)
                       for x1, y1, x2, y2 in person_boxes]
        confirmed = []
        for box in boxes:
            match = max(self.candidates, key=lambda c: _iou(c["box"], box), default=None)
            if match is not None and _iou(match["box"], box) >= 0.3:
                match.update(box=box, last=self.frame_no)
            else:
                match = {"box": box, "first": self.frame_no, "last": self.frame_no, "reported": False}
                self.candidates.append(match)
            if (not match["reported"] and self.frame_no - match["first"] >= self.persist_frames
                    and not any(_iou(box, area) > 0 for area in near_person)):
                match["reported"] = True
                kind = self._kind(box)
                confirmed.append({"box": tuple(map(int, box)), "first_frame": match["first"],
                                  "confirmed_frame": self.frame_no, "kind": kind, "edge_ratio": self.edge_ratio})
        # 1초 넘게 안 보인 후보는 버린다(끊기면 처음부터 다시 센다)
        self.candidates = [c for c in self.candidates if self.frame_no - c["last"] <= max(1, round(self.fps))]
        return confirmed

    def _kind(self, box):
        """영역 테두리의 윤곽선이 지금 화면에서 더 강하면 새로 놓인 물건, 장기 배경에서 더 강했으면 치워진 자리.

        놓인 물건은 바닥·선반 위에 새 윤곽을 만들고, 치워진 물건은 배경에 있던 윤곽이 사라진다.
        """
        pad = 2
        h, w = self._gray.shape
        x1, y1 = max(0, int(box[0] * SCALE) - pad), max(0, int(box[1] * SCALE) - pad)
        x2, y2 = min(w, int(box[2] * SCALE) + pad + 1), min(h, int(box[3] * SCALE) + pad + 1)
        border = cv2.morphologyEx(self._mask[y1:y2, x1:x2], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        self.edge_ratio = 1.0
        if not border.any():
            return "added"

        def edge(image):
            patch = image[y1:y2, x1:x2]
            return float(np.hypot(cv2.Sobel(patch, cv2.CV_32F, 1, 0), cv2.Sobel(patch, cv2.CV_32F, 0, 1))[border].mean())

        self.edge_ratio = edge(self._gray) / (edge(self.long_bg) + 1e-6)   # 분류기 특징용으로 남긴다
        return "added" if self.edge_ratio >= NEW_OBJECT_EDGE_RATIO else "removed"
