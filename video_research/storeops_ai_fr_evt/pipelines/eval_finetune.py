"""미세조정 S3D(4클래스) OOF 평가: 행동별 이벤트 단위 재현율·오경보/시간, 창 1개 기준(경보 = 점수 >= thr인 연속 창 구간 하나).

점수 = 그 행동 로짓 - logsumexp(나머지 로짓)(그 행동일 로그 오즈). 보류 fold의 모든 창을 쓰며 공식 validation은 읽지 않는다.
- 클립 적중: 그 행동 클립에서 경보 구간이 양성(그 행동 라벨 1) 창을 포함. 오경보: 경보 구간에 그 행동 양성도 애매(-1)도 없음.
  다른 행동 사건 구간에서 난 경보도 그 행동의 오경보로 센다(서비스는 그 카테고리를 낼 것이기 때문).
- 오경보/시간 = 오경보 구간 수 / 음성 시간. 구간 수는 '항상 경보'가 유리해지는 허점(경보 구간 하나로 합쳐짐)이 있어서, 기준 선택은 음성 창 중 경보가 울린 창 비율(fa_windows_per_hour)로 한다.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

from pipelines.aihub_abandonment import APP_ROOT
from pipelines.oof_search import candidates_from_values

CLASSES = {1: "전도", 2: "유기", 3: "절도"}
STRIDE_SEC = 2.0


def oof_files() -> list[str]:
    """EVAL_DIR(기본 finetune_s3d = A안; finetune_s3d_c1 = C1)의 OOF 파일. EVAL_FOLDS="0,1"처럼 fold를 골라 같은 fold끼리 비교할 수 있다."""
    folder = APP_ROOT / "output" / os.getenv("EVAL_DIR", "finetune_s3d")
    keep = {x for x in os.getenv("EVAL_FOLDS", "").split(",") if x}
    return [f for f in sorted(glob.glob(str(folder / "oof_fold*.npz"))) if not keep or Path(f).stem.replace("oof_fold", "") in keep]


def load() -> dict[str, np.ndarray]:
    parts = [np.load(f) for f in oof_files()]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("clip", "start", "label", "cls", "behavior", "logits")}


def log_odds(logits: np.ndarray, c: int) -> np.ndarray:
    rest = np.delete(logits, c, axis=1)
    m = rest.max(1)
    return logits[:, c] - (m + np.log(np.exp(rest - m[:, None]).sum(1)))


def per_clip(d: dict[str, np.ndarray], c: int) -> list[dict]:
    z = log_odds(d["logits"], c)
    out = []
    for clip in np.unique(d["clip"]):
        m = d["clip"] == clip
        order = np.argsort(d["start"][m])
        beh = d["behavior"][m][0]
        lab = d["label"][m][order]
        # 이 행동 기준 라벨: 이 행동 클립의 1 = 양성, 다른 행동 클립의 1 = 이 행동에게는 음성(0), -1 = 애매
        own = beh == CLASSES[c]
        # 다른 행동의 사건 구간(lab==1)은 기본적으로 이 행동의 음성(0)이다(엄격). IGNORE_OTHER_EVENTS=1이면 판정에서 제외(-1)해 '진짜 정상 구간'의 오경보만 센다.
        other = -1 if os.getenv("IGNORE_OTHER_EVENTS", "0") == "1" else 0
        y = np.where(lab == 1, 1 if own else other, lab)
        out.append({"clip": clip, "own": own, "z": z[m][order], "y": y})
    return out


def runs(z: np.ndarray, thr: float) -> list[tuple[int, int]]:
    hit = np.concatenate([[False], z >= thr, [False]])
    edges = np.flatnonzero(hit[1:] != hit[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def evaluate(clips: list[dict], thr: float) -> dict:
    hits = events = false = 0
    for c in clips:
        r = runs(c["z"], thr)
        if (c["y"] == 1).any():
            events += 1
            hits += any((c["y"][a:b + 1] == 1).any() for a, b in r)
        false += sum(1 for a, b in r if not (c["y"][a:b + 1] != 0).any())
    hours = sum(int((c["y"] == 0).sum()) for c in clips) * STRIDE_SEC / 3600
    fa_windows = sum(int(((c["z"] >= thr) & (c["y"] == 0)).sum()) for c in clips)  # 음성 창 중 경보가 울린 창 수(전 구간 경보를 이득으로 보지 않도록 시간 기준으로도 센다)
    return {"recall": hits / max(events, 1), "events": events, "false": int(false), "fa_per_hour": false / hours, "hours": hours,
            "fa_windows_per_hour": fa_windows / hours}


def main() -> None:
    d = load()
    folds = len(oof_files())
    print(f"보류 fold {folds}개, 창 {len(d['label'])}개")
    for c, name in CLASSES.items():
        clips = per_clip(d, c)
        pos = np.concatenate([x["z"][x["y"] == 1] for x in clips])
        auc_like = float(np.mean([x["z"][x["y"] == 1].max() > np.median(np.concatenate([y["z"][y["y"] == 0] for y in clips])) for x in clips if (x["y"] == 1).any()]))
        cand = candidates_from_values(np.concatenate([pos, np.concatenate([x["z"][x["y"] == 0] for x in clips])]), 120)
        print(f"\n[{name}] 양성 창 {len(pos)}개, 사건 클립 {sum((x['y'] == 1).any() for x in clips)}개")
        for target in (0.90, 0.95):
            rows = [(t, evaluate(clips, t)) for t in cand]
            ok = [(t, r) for t, r in rows if r["recall"] >= target]
            if not ok:
                print(f"  재현율>={target}: 만족 없음 (최대 재현율 {max(r['recall'] for _, r in rows):.3f})"); continue
            t, r = min(ok, key=lambda tr: (tr[1]["fa_windows_per_hour"], tr[1]["fa_per_hour"], -tr[1]["recall"]))
            print(f"  재현율>={target}: 기준 {t:.2f} 재현율 {r['recall']:.3f} | 오경보 구간 {r['false']}건 = {r['fa_per_hour']:.2f}/시간 | 음성 창 경보 {r['fa_windows_per_hour']:.2f}창/시간 (음성 {r['hours']:.1f}시간)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
