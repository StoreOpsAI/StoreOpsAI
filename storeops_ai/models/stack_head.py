"""A(화면 전체) + C1(사람 크롭) S3D 로짓을 학습된 스택(표준화 + 로지스틱)으로 결합한다.

계수·임계값은 연구 쪽 export_stack.py가 OOF로 정해 stack.json에 저장한다. 크롭 규칙(CROP_PAD, MIN_SIDE_RATIO)은 C1 학습 때와 같아야 한다.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

CROP_PAD = 0.5          # 사람 박스 합집합을 사방 폭·높이의 50%만큼 넓힌다(C1 학습과 동일)
MIN_SIDE_RATIO = 0.25   # 정사각형 한 변은 화면 높이의 25% 이상(작은 사람을 과도하게 확대하지 않음)


def union_crop(frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """창 안 사람 박스 합집합을 넓힌 정사각형으로 자른다. 박스가 없으면 화면 전체(학습과 같은 규칙)."""
    if not len(boxes):
        return frame
    h, w = frame.shape[:2]
    b = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = max((x2 - x1) * (1 + 2 * CROP_PAD), (y2 - y1) * (1 + 2 * CROP_PAD), MIN_SIDE_RATIO * h)
    side = min(side, float(min(h, w)))
    cx = float(np.clip((x1 + x2) / 2, side / 2, w - side / 2))
    cy = float(np.clip((y1 + y2) / 2, side / 2, h - side / 2))
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    return frame[y0:y0 + int(side), x0:x0 + int(side)]


def largest_person_boxes(per_frame: Sequence[Sequence[tuple]]) -> List[Sequence[float]]:
    """창 안 프레임마다 [(추적 번호, 박스), ...]에서 박스 면적 평균이 가장 큰 사람의 박스들만 돌려준다(MVP: 가장 큰 인물 기준, 카메라 하나)."""
    area: Dict[int, List[float]] = {}
    for frame in per_frame:
        for tid, b in frame:
            area.setdefault(int(tid), []).append(float((b[2] - b[0]) * (b[3] - b[1])))
    if not area:
        return []
    best = max(area, key=lambda t: float(np.mean(area[t])))
    return [b for frame in per_frame for tid, b in frame if int(tid) == best]


class StackHead:
    def __init__(self, path: str | Path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.classes: Dict[str, dict] = spec["classes"]
        # 행동별 과거 m개 창 점수 평균(인과적, 카메라 하나 기준 상태). m=1이면 평활 없음. 영상·스트림이 바뀌면 reset() 한다.
        self._hist: Dict[str, deque] = {cat: deque(maxlen=int(p.get("smooth", 1))) for cat, p in self.classes.items()}

    def reset(self) -> None:
        for h in self._hist.values():
            h.clear()

    def thresholds(self) -> Dict[str, float]:
        """카테고리별 임계값(확률 = sigmoid(로그 오즈 임계값))."""
        return {c: float(1 / (1 + np.exp(-p["threshold_logit"]))) for c, p in self.classes.items()}

    def level(self, category: str, prob: float) -> str | None:
        """위험도 단계(정상/주의/경고/높음). 경계는 OOF 동작점(재현율 0.95 / 0.90 / 오경보 예산 이하)에서 정한 값이다."""
        spec = self.classes.get(category)
        if spec is None or "levels" not in spec:
            return None
        z = float(np.log(max(prob, 1e-12) / max(1.0 - prob, 1e-12)))
        lv = spec["levels"]
        return "높음" if z >= lv["높음"] else "경고" if z >= lv["경고"] else "주의" if z >= lv["주의"] else "정상"

    @staticmethod
    def _log_odds(logits: np.ndarray, c: int) -> float:
        rest = np.delete(logits, c)
        m = rest.max()
        return float(logits[c] - (m + np.log(np.exp(rest - m).sum())))

    def _score(self, p: dict, la: np.ndarray, lc: np.ndarray) -> float:
        """행동별 점수(로그 오즈). 로지스틱 스택(계수가 있을 때) 또는 동적 결합(확신 큰 쪽에 비중)."""
        if "coef" in p:
            x = np.concatenate([la, lc]).astype(np.float64)
            return float(((x - p["mean"]) / p["std"]) @ np.array(p["coef"]) + p["bias"])
        a, b = self._log_odds(la, p["index"]), self._log_odds(lc, p["index"])
        return (a * abs(a) + b * abs(b)) / (abs(a) + abs(b) + 1e-9)

    def probs(self, logits_a: np.ndarray, logits_c1: np.ndarray) -> Dict[str, float]:
        la, lc = np.asarray(logits_a, dtype=np.float64), np.asarray(logits_c1, dtype=np.float64)
        out = {}
        for cat, p in self.classes.items():
            self._hist[cat].append(self._score(p, la, lc))
            out[cat] = float(1 / (1 + np.exp(-float(np.mean(self._hist[cat])))))
        return out


def _self_check() -> None:
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert union_crop(frame, []).shape == frame.shape                       # 사람이 없으면 화면 전체
    c = union_crop(frame, [[100, 200, 160, 400], [300, 220, 360, 420]])
    assert c.shape[0] == c.shape[1] >= int(0.25 * 1080) - 1 and c.shape[0] <= 1080   # 정사각형, 최소 크기
    import tempfile
    d = {"classes": {"쓰러짐": {"mean": [0.0] * 8, "std": [1.0] * 8, "coef": [1.0] + [0.0] * 7, "bias": 0.0, "threshold_logit": 2.0}}}
    f = Path(tempfile.mkdtemp()) / "s.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    s = StackHead(f)
    assert abs(s.probs(np.array([2.0, 0, 0, 0]), np.zeros(4))["쓰러짐"] - 1 / (1 + np.exp(-2.0))) < 1e-9
    dyn = {"classes": {"쓰러짐": {"index": 1, "method": "dynamic", "threshold_logit": 2.0}}}
    f2 = f.with_name("d.json"); f2.write_text(json.dumps(dyn), encoding="utf-8")
    sd = StackHead(f2)
    la, lc = np.array([0.0, 5.0, 0, 0]), np.array([0.0, 5.0, 0, 0])
    assert abs(sd.probs(la, lc)["쓰러짐"] - 1 / (1 + np.exp(-(5 - np.log(3))))) < 1e-9 and abs(sd.thresholds()["쓰러짐"] - 1 / (1 + np.exp(-2.0))) < 1e-9
    d['classes']['쓰러짐']['levels'] = {'주의': 1.0, '경고': 2.0, '높음': 4.0}
    f.write_text(json.dumps(d), encoding='utf-8')
    s = StackHead(f)
    assert [s.level('쓰러짐', 1 / (1 + np.exp(-z))) for z in (0.5, 1.5, 3.0, 5.0)] == ['정상', '주의', '경고', '높음'] and s.level('절도', 0.9) is None
    assert abs(s.thresholds()["쓰러짐"] - 1 / (1 + np.exp(-2.0))) < 1e-9
    pf = [[(1, [0, 0, 10, 20]), (2, [0, 0, 100, 200])], [(2, [0, 0, 110, 210])], [(1, [0, 0, 12, 22])]]
    assert largest_person_boxes(pf) == [[0, 0, 100, 200], [0, 0, 110, 210]] and largest_person_boxes([]) == []
    sm = {"classes": {"쓰러짐": {"index": 1, "method": "dynamic", "smooth": 2, "threshold_logit": 0.0}}}
    f3 = f.with_name("sm.json"); f3.write_text(json.dumps(sm), encoding="utf-8")
    h = StackHead(f3)
    hi, lo = np.array([0.0, 8.0, 0, 0]), np.array([0.0, -8.0, 0, 0])
    p1, p2 = h.probs(hi, hi)["쓰러짐"], h.probs(lo, lo)["쓰러짐"]
    a_hi, a_lo = 8 - np.log(3), -8 - np.log(3)    # 로그 오즈: (8) - ln3, (-8) - ln3
    assert p1 > 0.99 and abs(p2 - 1 / (1 + np.exp(-(a_hi + a_lo) / 2))) < 1e-9     # 두 창 점수의 평균이 확률로 환산된다
    h.reset(); assert h.probs(lo, lo)["쓰러짐"] < 0.01
    print("ok")


if __name__ == "__main__":
    _self_check()
