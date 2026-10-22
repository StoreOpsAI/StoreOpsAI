"""손-물건 접촉 탐지기(Shan, Geng, Shu, Fouhey, CVPR 2020, 100DOH 100K 가중치) 실시간 래퍼와 사람별 손 상태.

- 탐지는 별도 스레드에서 가장 최근 프레임만 돌린다(A4000 약 54ms/장, 2026-10-21 측정). 메인 루프는 기다리지 않는다.
- 손 상태: 손 상자 중심이 사람 박스 안이면 그 사람의 손(여럿이면 가장 작은 박스). 사람 단위로 본다: 한 장에서 쥔 손(P)이 하나라도
  있으면 쥠, 손이 보이는데 모두 접촉 없음(N)이면 빈손. 좌우 손은 따로 잇지 않는다(물건을 다른 손으로 옮기거나 좌우 판단이 바뀌어
  run27 E008 '들고 나감'이 숨김으로 잘못 나옴, 2026-10-22).
- 숨김 후보(근거 약함, 행동인식_분석 11절): 그 사람이 since 이후 쥔 손을 보였고, 마지막 쥠 뒤 빈손 장이 있었다.
  손이 안 보이는 장(44장 중 4~13장만 쥔 손이 잡힘)은 빈손으로 치지 않는다.
- 저장소 코드는 nms·roi_align·roi_pool만 torchvision.ops로 바꿔 HAND_OBJECT_DIR/repo에 둔다(MIT). 가중치가 없으면 예외.
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

STATES = "NSOPF"   # 접촉 없음 / 자기 몸 / 다른 사람 / 들 수 있는 물건 / 고정 가구
EMPTY, HELD = 0, 3
WEIGHTS = "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_89999.pth"


def touch(a, b) -> bool:
    """두 상자 (x1, y1, x2, y2)가 닿거나 겹치는가."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


class HandObjectDetector:
    def __init__(self, root: str, conf: float):
        self.conf = conf
        self.latest = None   # (t, hands[(box, state, side)], objs[box])
        self._job = None
        self._cv = threading.Condition()
        self._load(Path(root))
        threading.Thread(target=self._loop, daemon=True).start()

    def _load(self, root: Path):
        repo = root / "repo"
        if not (repo / WEIGHTS).is_file():
            raise FileNotFoundError(f"손-물건 탐지기 가중치가 없습니다: {repo / WEIGHTS}")
        for p in (root / "pydeps", repo / "lib"):
            if str(p) not in sys.path:
                sys.path.append(str(p))   # 뒤에 붙여 우리 모듈(utils, config 등)을 가리지 않는다
        import torch
        from model.faster_rcnn.resnet import resnet
        from model.roi_layers import nms
        from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
        from model.utils.config import cfg, cfg_from_file
        cfg_from_file(str(repo / "cfgs/res101.yml"))
        cfg.CUDA = True
        net = resnet(np.asarray(["__background__", "targetobject", "hand"]), 101, pretrained=False, class_agnostic=False)
        net.create_architecture()
        ck = torch.load(repo / WEIGHTS, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["model"])
        cfg.POOLING_MODE = ck["pooling_mode"]
        net.cuda()
        net.eval()   # 저장소 resnet.train()은 self를 돌려주지 않아 이어 쓰면 None이 된다
        self.net = net
        self.torch, self.cfg, self.nms, self.inv, self.clip = torch, cfg, nms, bbox_transform_inv, clip_boxes
        self.std = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
        self.mean = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

    def submit(self, frame, t: float):
        with self._cv:
            self._job = (frame, t)
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                frame, t = self._job
                self._job = None
            try:
                self.latest = (t, *self.detect(frame))
            except Exception as exc:   # 탐지 한 장 실패로 시연 전체를 멈추지 않는다(손 상태만 비게 됨)
                print(f"[손-물건 탐지 실패] {exc}")

    def detect(self, im):
        """저장소 demo.py와 같은 전처리·후처리. 반환: 손 [(상자, 접촉, 좌우)], 물건 [상자]."""
        torch, cfg = self.torch, self.cfg
        x = (im.astype(np.float32) - cfg.PIXEL_MEANS).astype(np.float32)   # PIXEL_MEANS가 float64라 그대로 빼면 float64가 된다
        s = min(cfg.TEST.SCALES[0] / min(x.shape[:2]), cfg.TEST.MAX_SIZE / max(x.shape[:2]))
        x = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        data = torch.from_numpy(x).permute(2, 0, 1)[None].contiguous().cuda()
        info = torch.tensor([[x.shape[0], x.shape[1], s]], dtype=torch.float32).cuda()
        z = torch.zeros(1, 1, 5).cuda()
        with torch.no_grad():
            rois, prob, bp, *_, ll = self.net(data, info, z, torch.zeros(1, dtype=torch.long).cuda(), z)
        sc = prob[0]
        deltas = (bp[0].view(-1, 4) * self.std + self.mean).view(1, -1, 12)
        boxes = self.clip(self.inv(rois[0:1, :, 1:5], deltas, 1), info, 1)[0] / s
        contact, side = ll[0][0][0].argmax(-1), (torch.sigmoid(ll[2][0][0, :, 0]) > 0.5).long()
        out = []
        for j in (2, 1):   # 손, 물건
            idx = torch.nonzero(sc[:, j] > self.conf).view(-1)
            bx = boxes[idx, j * 4:(j + 1) * 4]
            keep = self.nms(bx, sc[idx, j], cfg.TEST.NMS).tolist() if idx.numel() else []
            out.append([(tuple(bx[k].tolist()), int(contact[idx[k]]), int(side[idx[k]])) if j == 2 else tuple(bx[k].tolist()) for k in keep])
        return out[0], out[1]


class HandState:
    """사람별·손별 마지막 쥠/빈손 시각과 최근 탐지 기록(horizon 초)."""

    def __init__(self, horizon: float):
        self.horizon = horizon
        self.held, self.empty = {}, {}
        self.recent = deque()   # (t, [(손 상자, 접촉)], [물건 상자])
        self._last_t = None

    @staticmethod
    def owner_of(box, people: dict):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = [(abs((b[2] - b[0]) * (b[3] - b[1])), tid) for tid, b in people.items() if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        return min(inside)[1] if inside else None

    def update(self, result, people: dict):
        if result is None or result[0] == self._last_t:
            return
        t, hands, objs = result
        self._last_t = t
        by_person = {}
        for box, state, _ in hands:
            tid = self.owner_of(box, people)
            if tid is not None:
                by_person.setdefault(tid, []).append(state)
        for tid, states in by_person.items():
            if HELD in states:
                self.held[tid] = t
            elif EMPTY in states and tid in self.held:   # 손이 보이는데 쥔 손이 하나도 없다
                self.empty[tid] = t
        self.recent.append((t, [(b, st) for b, st, _ in hands], objs))
        while self.recent and self.recent[0][0] < t - self.horizon:
            self.recent.popleft()

    def concealed(self, tid, since: float) -> bool:
        """그 사람이 since 이후 쥔 손을 보였고, 마지막 쥠 뒤에 '손이 보이는데 쥔 손이 없는' 장이 있었는가."""
        h = self.held.get(tid, -1e9)
        return h >= since and self.empty.get(tid, -1e9) > h

    def touched(self, spot, since: float) -> bool:
        """since 이후 쥔 손(P)이나 그 장의 물건 상자가 자리 spot에 닿은 장이 있는가(2026-10-21 오프라인 시험과 같은 규칙)."""
        for t, hands, objs in self.recent:
            held = [b for b, st in hands if st == HELD]
            if t >= since and held and any(touch(b, spot) for b in held + list(objs)):
                return True
        return False


def draw_hands(out, result):
    """손 상자(쥠=주황, 빈손·기타=하늘)와 물건 상자(노랑)를 그린다."""
    if result is None:
        return
    _, hands, objs = result
    for box, state, side in hands:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 140, 255) if state == HELD else (255, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{'RL'[side == 0]}-{STATES[state]}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for box in objs:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)


def _self_check():
    hs = HandState(horizon=30.0)
    people = {1: (0, 0, 100, 300), 2: (200, 0, 300, 300)}
    hs.update((1.0, [((40, 100, 60, 120), HELD, 1)], [(45, 110, 70, 140)]), people)
    assert hs.touched((65, 130, 80, 150), 0.0) and not hs.touched((150, 130, 160, 150), 0.0)
    assert not hs.concealed(1, 0.0)
    hs.update((2.0, [((40, 100, 60, 120), EMPTY, 0), ((60, 100, 80, 120), HELD, 1)], []), people)   # 한 손은 빈손, 다른 손은 쥠: 숨김 아님
    assert not hs.concealed(1, 0.0)
    hs.update((2.5, [], []), people)   # 손이 안 보임(놓친 장): 빈손으로 치지 않는다
    assert not hs.concealed(1, 0.0)
    hs.update((3.0, [((40, 100, 60, 120), EMPTY, 1)], []), people)   # 보이는 손이 모두 빈손: 숨김 후보
    assert hs.concealed(1, 0.0) and not hs.concealed(1, 2.5) and not hs.concealed(2, 0.0)
    hs.update((3.0, [((40, 100, 60, 120), HELD, 1)], []), people)   # 같은 시각 결과는 다시 반영하지 않는다
    assert hs.concealed(1, 0.0)
    print("hand_object self-check ok")


if __name__ == "__main__":
    _self_check()
