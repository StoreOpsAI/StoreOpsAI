"""abandon_cache의 사람 추적·물건 탐지 원본으로 사람별 시간 창의 사람-물건 관계 시계열을 만든다 (임계값 없이 헤드가 학습).

근거: Chao 외 WACV 2018(HICO-DET)은 사람·물건 박스의 공간 관계를 학습망 입력으로 쓰고, Shan 외 CVPR 2020은 손의 물건 접촉 상태를
학습으로 추정한다. 여기서는 프레임마다 손목에 가장 가까운 물건 박스 하나를 사람 박스 기준 좌표로 바꿔 채널로 쓰고
(쥠·놓음·정지·멀어짐은 시계열에서 헤드가 배운다), 같은 추적의 등장 직후 SEQ_ENTRY_FRAMES 프레임도 덧붙인다.
쥠/접촉 프레임 라벨이 없어서 접촉은 사건 라벨로 약하게 감독된다(논문과 다른 점).
"""
from __future__ import annotations

import numpy as np

from config.config import ABANDON_WIN_FRAMES, SEQ_ENTRY_FRAMES

CHANNELS = ("person", "obj", "ox1", "oy1", "ox2", "oy2", "conf", "wl_dx", "wl_dy", "wr_dx", "wr_dy", "wl_c", "wr_c", "pdx", "pdy")
CLIP_RANGE = 5.0


def track_vectors(trk: np.ndarray, det: np.ndarray, tid: int) -> dict[int, np.ndarray]:
    """한 추적의 프레임별 관계 벡터(len(CHANNELS))."""
    rows = trk[trk[:, 0] == tid]
    out, prev = {}, None
    for r in rows[np.argsort(rows[:, 1])]:
        f = int(r[1])
        x1, y1, x2, y2 = r[2:6]
        h = max(y2 - y1, 1.0)
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        wl, wr = r[6:8], r[9:11]
        v = np.zeros(len(CHANNELS), np.float32)
        v[0] = 1.0
        v[11], v[12] = r[8], r[11]
        cand = det[det[:, 0] == f]
        if len(cand):
            centers = np.stack([(cand[:, 1] + cand[:, 3]) / 2, (cand[:, 2] + cand[:, 4]) / 2], axis=1)
            d = np.minimum(np.hypot(*(centers - wl).T), np.hypot(*(centers - wr).T))
            k = int(d.argmin())
            ox1, oy1, ox2, oy2, conf = cand[k, 1:6]
            v[1], v[2:6], v[6] = 1.0, [(ox1 - c[0]) / h, (oy1 - c[1]) / h, (ox2 - c[0]) / h, (oy2 - c[1]) / h], conf
            v[7:9], v[9:11] = (wl - centers[k]) / h, (wr - centers[k]) / h
        if prev is not None and f - prev[0] == 1:
            v[13:15] = (c - prev[1]) / h
        prev = (f, c)
        out[f] = np.clip(v, -CLIP_RANGE, CLIP_RANGE)
    return out


def window_sequences(cache) -> tuple[np.ndarray, np.ndarray]:
    """창마다 (W, C) 시계열과 등장 직후 (E, C) 시계열. 반환: (N, W, C), (N, E, C)."""
    trk, det = cache["trk"], cache["det"]
    vectors = {int(t): track_vectors(trk, det, int(t)) for t in set(cache["win_tid"].tolist())}
    n, c = len(cache["win_tid"]), len(CHANNELS)
    win = np.zeros((n, ABANDON_WIN_FRAMES, c), np.float32)
    entry = np.zeros((n, SEQ_ENTRY_FRAMES, c), np.float32)
    for i, (t, s) in enumerate(zip(cache["win_tid"], cache["win_start"])):
        vec = vectors[int(t)]
        for k in range(ABANDON_WIN_FRAMES):
            if int(s) + k in vec:
                win[i, k] = vec[int(s) + k]
        for k, f in enumerate(sorted(vec)[:SEQ_ENTRY_FRAMES]):
            entry[i, k] = vec[f]
    return win, entry


def _self_check():
    """손목이 물건 위에 있다가(거리 0) 떠나면 wl_dx가 0에서 커지고, 물건이 없으면 obj=0."""
    trk = np.array([[1, f, 0, 0, 40, 100, 20 + 10 * max(f - 3, 0), 50, 0.9, 20 + 10 * max(f - 3, 0), 50, 0.9] for f in range(6)], np.float32)
    det = np.array([[f, 15, 45, 25, 55, 0.8] for f in range(6)], np.float32)
    cache = {"trk": trk, "det": det, "win_tid": np.array([1]), "win_start": np.array([0])}
    win, entry = window_sequences(cache)
    ix = CHANNELS.index("wl_dx")
    assert abs(win[0, 0, ix]) < 0.2 and win[0, 5, ix] > 0.15, (win[0, 0, ix], win[0, 5, ix])
    assert window_sequences({**cache, "det": np.zeros((0, 6), np.float32)})[0][0, 0, CHANNELS.index("obj")] == 0


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
