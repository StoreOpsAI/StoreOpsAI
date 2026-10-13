"""손 크롭 시험 2단계: 창마다 세 가지 입력(화면 전체 A / 사람 크롭 C1 / 손 크롭 H)의 얼린 S3D(Kinetics) 특징을 뽑고, 날짜 fold 선형 프로브로 유기·절도 분리 성능을 비교한다.

- 손 크롭 H: 프레임마다 손목(COCO 9·10, 신뢰도>=WRIST_CONF)이 보이는 가장 큰 사람의 손목 중점을 따라가는 한 변 FIXED_SIDE px 정사각형(두 손목이 멀면 넓힘, 중심은 EMA 평활)을 원본 해상도에서 잘라 224로 줄인다.
  (창 합집합 방식은 걷는 동작에서 크롭이 화면 전체로 커져 확대 효과가 없어 폐기: 한 변 861~1080px) 포즈 캐시: output/pose_cache/<행동>/train/<영상>.npz.
- 특징: torchvision S3D(Kinetics-400 사전학습) features 평균 풀링 1024차원, 입력은 12프레임 -> 24프레임, 서비스와 같은 정규화.
- 비교: 입력별 로지스틱 프로브(날짜 fold CV, 클래스 균형) OOF 점수 -> 창 단위 AUROC와 이벤트 단위(재현율 0.9에서 오경보/시간). 조합은 로그 오즈 평균.
- 사용 영상: 포즈 캐시가 있는 격리되지 않은 학습 영상. 공식 validation은 쓰지 않는다. 사용: python -m pipelines.hand_pilot_probe [--extract] [--eval]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from config.config import BEHAVIOR_FEATURE_ROOT
from pipelines import finetune_s3d as F
from pipelines.fullscene_windows import MEAN, NUM_FRAMES, SIZE, STD, STRIDE, WIN, video_path
from pipelines.video_sampling import is_sample

FEAT = Path(BEHAVIOR_FEATURE_ROOT) / "hand_pilot"
POSE = F.APP_ROOT / "output" / "pose_cache"
WRIST_CONF, MARGIN, FIXED_SIDE, EMA_ALPHA = 0.3, 150, 400, 0.5


def frame_centers(kpts: dict[int, np.ndarray], n: int, shape: tuple[int, int]) -> list[tuple[float, float, int]]:
    """표본마다 (중심 x, 중심 y, 한 변). 손목(신뢰도>=WRIST_CONF)이 보이는 사람 중 박스가 가장 큰 사람의 손목 중점을 중심으로, 두 손목이 멀면 그만큼 넓힌다.
    손목이 없는 프레임은 직전 중심을 유지하고, 중심은 지수 평활(EMA_ALPHA)로 떨림을 줄인다. 한 변은 FIXED_SIDE(두 손목 거리+2*MARGIN이 더 크면 그만큼), 화면 높이 이하."""
    h, w = shape
    out, prev = [], (w / 2, h / 2)
    for t in range(n):
        best = None
        for k in kpts.get(t, []):
            pts = [k[j][:2] for j in (9, 10) if k[j][2] >= WRIST_CONF]
            if pts:
                area = (k[:, 0].max() - k[:, 0].min()) * (k[:, 1].max() - k[:, 1].min())
                if best is None or area > best[0]:
                    best = (area, np.array(pts))
        side = FIXED_SIDE
        if best is not None:
            p = best[1]
            c = p.mean(0)
            side = max(FIXED_SIDE, int(np.ptp(p, axis=0).max() + 2 * MARGIN))
            prev = (EMA_ALPHA * c[0] + (1 - EMA_ALPHA) * prev[0], EMA_ALPHA * c[1] + (1 - EMA_ALPHA) * prev[1])
        side = int(min(side, h))
        out.append((float(np.clip(prev[0], side / 2, w - side / 2)), float(np.clip(prev[1], side / 2, h - side / 2)), side))
    return out


def hand_windows(video: Path, kpts: dict[int, np.ndarray], starts: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    fps, raw, t = cap.get(cv2.CAP_PROP_FPS), 0, 0
    centers, out = None, {s: np.zeros((WIN, SIZE, SIZE, 3), np.uint8) for s in starts}
    n_all = max((s + WIN for s in starts), default=0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if not is_sample(raw - 1, fps):
            continue
        if centers is None:
            centers = frame_centers(kpts, n_all, frame.shape[:2])
        cx, cy, side = centers[t] if t < len(centers) else centers[-1]
        x, y = int(cx - side / 2), int(cy - side / 2)
        crop = cv2.resize(frame[y:y + side, x:x + side], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        for s in starts:
            if s <= t < s + WIN:
                out[s][t - s] = crop
        t += 1
    cap.release()
    return out


def embed_windows(model, wins: list[np.ndarray], device: str) -> np.ndarray:
    idx = np.round(np.linspace(0, WIN - 1, NUM_FRAMES)).astype(int)
    out = []
    for i in range(0, len(wins), 8):
        batch = [((w[idx][..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(3, 0, 1, 2) for w in wins[i:i + 8]]
        with torch.inference_mode():
            out.append(model(torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(device)).mean(dim=(2, 3, 4)).float().cpu().numpy())
    return np.concatenate(out)


def jpeg_windows(items: list[list[bytes]]) -> list[np.ndarray]:
    return [np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in jp]) for jp in items]


def extract() -> None:
    from torchvision.models.video import S3D_Weights, s3d

    device = "cuda"
    model = s3d(weights=S3D_Weights.DEFAULT).features.to(device).eval()
    FEAT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in F.build_index()]
    clips = sorted({(r["clip"], r["behavior"]) for r in rows if (POSE / r["behavior"] / "train" / f"{r['clip']}.npz").is_file()})
    print(f"대상 영상 {len(clips)}개", flush=True)
    for k, (clip, beh) in enumerate(clips, 1):
        target = FEAT / f"{clip}.npz"
        if target.is_file():
            continue
        z = np.load(F.FULL / f"{clip}.npz")
        starts = z["start"].tolist()
        pz = np.load(POSE / beh / "train" / f"{clip}.npz")
        kpts: dict[int, np.ndarray] = {}
        for fr, kp in zip(pz["frame"], pz["kpt"]):
            kpts.setdefault(int(fr), []).append(kp)
        kpts = {f: np.stack(v) for f, v in kpts.items()}
        data = pickle.loads((F.FRAMES / f"{clip}.pkl").read_bytes())["jpeg"]
        fa = embed_windows(model, jpeg_windows([data[s:s + WIN] for s in starts]), device)
        crops = pickle.loads((F.FEATURES / "crops_c1" / f"{clip}.pkl").read_bytes())
        fb = embed_windows(model, jpeg_windows([crops[s] for s in starts]), device)
        hw = hand_windows(video_path(clip, beh, "train"), kpts, starts)
        fc = embed_windows(model, [hw[s] for s in starts], device)
        np.savez_compressed(target, start=np.array(starts), fa=fa, fb=fb, fc=fc)
        if k % 20 == 0:
            print(f"[{k}/{len(clips)}] {clip}", flush=True)
    print("HAND_EXTRACT_DONE", flush=True)


def evaluate_probe() -> None:
    from sklearn.metrics import roc_auc_score
    from pipelines.compare_backbones import _fit_linear
    from pipelines.eval_finetune import CLASSES, evaluate
    from pipelines.fusion_eval import clips_for
    from pipelines.oof_search import candidates_from_values

    lab = {(r["clip"], r["start"]): r for r in F.build_index()}
    keys, fa, fb, fc = [], [], [], []
    for f in sorted(FEAT.glob("*.npz")):
        z = np.load(f)
        for i, s in enumerate(z["start"].tolist()):
            if (f.stem, s) in lab:
                keys.append(lab[(f.stem, s)]); fa.append(z["fa"][i]); fb.append(z["fb"][i]); fc.append(z["fc"][i])
    d = {"clip": np.array([r["clip"] for r in keys]), "start": np.array([r["start"] for r in keys]), "label": np.array([r["label"] for r in keys]),
         "cls": np.array([r["cls"] for r in keys]), "behavior": np.array([r["behavior"] for r in keys]), "fold": np.array([r["fold"] for r in keys])}
    X = {"A 화면 전체": np.array(fa, np.float32), "C1 사람 크롭": np.array(fb, np.float32), "H 손 크롭": np.array(fc, np.float32)}
    valid, folds = d["label"] != -1, sorted(set(d["fold"].tolist()))
    print(f"창 {len(keys)}개, 영상 {len(set(d['clip']))}개(유기 {len(set(d['clip'][d['behavior'] == '유기']))}, 절도 {len(set(d['clip'][d['behavior'] == '절도']))}, 정상 {len(set(d['clip'][np.isin(d['behavior'], ['매장이동', '반품'])]))})", flush=True)
    for c, name in CLASSES.items():
        if c == 1:
            continue
        y = np.where(d["cls"] == c, 1, 0).astype(np.int64)
        zs = {}
        for k, x in X.items():
            z = np.zeros(len(y))
            for fo in folds:
                tr, te = valid & (d["fold"] != fo), d["fold"] == fo
                p = _fit_linear(x[tr], y[tr])(x[te])
                z[te] = np.log(p + 1e-6) - np.log(1 - p + 1e-6)
            zs[k] = z
        zs["A+C1 평균"] = (zs["A 화면 전체"] + zs["C1 사람 크롭"]) / 2
        zs["A+C1+H 평균"] = (zs["A 화면 전체"] + zs["C1 사람 크롭"] + zs["H 손 크롭"]) / 3
        zs["A+H 평균"] = (zs["A 화면 전체"] + zs["H 손 크롭"]) / 2
        print(f"\n[{name}] 양성 창 {int(((y == 1) & valid).sum())}개", flush=True)
        for k, z in zs.items():
            clips = clips_for(d, z, c)
            ok = [(t, evaluate(clips, t)) for t in candidates_from_values(z[valid], 100)]
            ok = [(t, r) for t, r in ok if r["recall"] >= 0.9]
            fa_s = f"{min(ok, key=lambda tr: (tr[1]['fa_windows_per_hour'], tr[1]['fa_per_hour']))[1]['fa_per_hour']:.2f}/시간" if ok else "재현율 0.9 불가"
            print(f"  {k:14s} AUROC {roc_auc_score(y[valid], z[valid]):.4f} | 재현율 0.9에서 오경보 {fa_s}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.eval:
        evaluate_probe()
