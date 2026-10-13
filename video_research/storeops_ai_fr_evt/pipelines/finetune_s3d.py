"""정상·전도·유기·절도 4클래스 S3D 미세조정(날짜 fold OOF): 화면 전체 4초 창, 서비스와 같은 입력 형식(224x224, 12프레임 -> 24프레임 균일 샘플링).

- 데이터: 전 행동 공통 분할표(data/global_split.csv)의 학습 가능 클립(격리 제외, 공식 validation 미사용), JPEG 프레임 캐시(frames224).
- 양성: 전도=영상의 '누운 시간 >= 3초' 창(fullscene_windows.window_label), 유기·절도=xml 사건 구간과 창이 절반 이상 겹침.
  행동마다 양성 창을 TARGET_POS개로 고정한다(행동 간 공정 비교):
  클립을 시드 고정 순서로 fold마다 TARGET_POS/5개 양성이 될 때까지 뽑는다(fold 소속은 그대로라 누수 없음).
- 음성(정상 클래스): 에폭마다 양성 총수만큼. 절반은 선택된 세 행동 영상의 사건 밖 구간(행동마다 같은 수), 절반은 정상 풀(반품·매장이동).
- 증강(학습 전용): 좌우 반전, 밝기·채도, 무작위 크롭 후 복원. 보류 fold에는 적용하지 않는다(AGENTS.md 3).
- 보류 fold: 모든 창(애매한 -1 포함)을 채점해 로짓(양성 출력 - 음성 출력)을 저장한다 -> output/finetune_s3d/oof_fold{k}.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config.config import ABANDON_EVENT_MARGIN, AIHUB_ORGANIZED_ROOT, BEHAVIOR_FEATURE_ROOT
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.behavior_labels import parse_label
from pipelines.fullscene_windows import MANIFEST_PATH
from pipelines.global_split import SPLIT_PATH

FEATURES = Path(BEHAVIOR_FEATURE_ROOT)
FULL = FEATURES / "fullscene"
INPUT_DIRS = {"full": "frames224", "c1": "crops_c1", "c2": "crops_c2", "h": "crops_h"}  # 입력 방식: 화면 전체 / 사람 중심 크롭 / 크롭 + 물건 박스 표시
INPUT = os.getenv("S3D_INPUT", "full")
FRAMES = FEATURES / INPUT_DIRS[INPUT]
PRE_EVENT_MARGIN = int(os.getenv("PRE_EVENT_MARGIN", "0"))   # >0이면 사건 시작 앞 이 표본(3fps) 안의 정상 창을 학습 정상에서 뺀다(채점에는 영향 없음). 0 = 기존 동작
POST_EVENT_MARGIN = int(os.getenv("POST_EVENT_MARGIN", "0"))   # >0이면 사건 끝 뒤 이 표본 안의 정상 창도 학습 정상에서 뺀다(채점에는 영향 없음). 0 = 기존 동작
AUDIT_NORMALS = os.getenv("AUDIT_NORMALS") == "1"   # 1이면 사건 영상의 사건 밖 창은 output/normal_audit_claude_*.csv에서 통과(pass)한 창만 정상으로 쓴다(Claude 시각 검수, 사람 검수 아님)
SITE_FOLDS = os.getenv("SITE_FOLDS") == "1"   # 1이면 fold = 촬영 장소(7곳): 장소 하나를 통째로 빼고 학습해 장소 지름길을 평가한다(--fold 0~6)
SIX = os.getenv("S3D_CLASSES", "4") == "6"   # 6이면 정상 + 전도·유기·절도·파손·방화 6클래스(반품은 정상 풀에서 뺀다). 결과는 *6 폴더에 따로 저장
OUT = APP_ROOT / "output" / (("finetune_s3d" if INPUT == "full" else f"finetune_s3d_{INPUT}") + ("6" if SIX else "") + (f"_pre{PRE_EVENT_MARGIN}" if PRE_EVENT_MARGIN else "") + (f"_post{POST_EVENT_MARGIN}" if POST_EVENT_MARGIN else "") + ("_audit" if AUDIT_NORMALS else "") + ("_site" if SITE_FOLDS else ""))
WIN, NUM_FRAMES, SIZE = 12, 24, 224
TARGET_POS = 3000  # 사용자 지정: 행동마다 양성 창 3,000개
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)
CLASS_OF = {"전도": 1, "유기": 2, "절도": 3}  # 0=정상. 서비스 label_map: normal, fall, abandon, theft
LABEL_MAP = {"normal": 0, "fall": 1, "abandon": 2, "theft": 3}
SOURCE = {"전도": "fall", "유기": "abandon", "절도": "theft", "반품": "normal", "매장이동": "normal"}
if SIX:
    CLASS_OF.update({"파손": 4, "방화": 5})
    LABEL_MAP.update({"break": 4, "fire": 5})
    SOURCE.update({"파손": "break", "방화": "fire"})
    del SOURCE["반품"]   # 사용자 지시(2026-10-08): 반품 영상은 쓰지 않는다


def event_window_labels(behavior: str, clip: str, starts: list[int], split: str = "train") -> list[int]:
    """유기·절도: 사건 구간과 겹침이 창의 절반 이상이면 1, 겹치거나 사건 앞뒤 ABANDON_EVENT_MARGIN 안이면 -1(제외), 그 밖은 0."""
    events, _ = parse_label(Path(AIHUB_ORGANIZED_ROOT) / behavior / split / "labels" / f"{clip}.xml", behavior)
    out = []
    for s0 in starts:
        best, near = 0, False
        for e in events:
            best = max(best, max(0, min(s0 + WIN, e["end"] + 1) - max(s0, e["start"])))
            near |= s0 < e["end"] + 1 + ABANDON_EVENT_MARGIN and s0 + WIN > e["start"] - ABANDON_EVENT_MARGIN
        out.append(1 if best >= WIN / 2 else (-1 if (best > 0 or near) else 0))
    return out


_FALLS: dict | None = None


def event_span(behavior: str, clip: str) -> tuple[int, int] | None:
    """그 영상의 (첫 사건 시작, 마지막 사건 끝) 표본. 전도는 매니페스트의 누운 구간, 나머지는 라벨 xml. 사건이 없으면 None."""
    global _FALLS
    if behavior == "전도":
        if _FALLS is None:
            m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            _FALLS = {c["clip"]: c["falls"] for part in m if isinstance(m[part], list) for c in m[part] if isinstance(c, dict) and "falls" in c}
        falls = _FALLS.get(clip)
        return (min(f["start"] for f in falls), max(f["end"] for f in falls)) if falls else None
    events, _ = parse_label(Path(AIHUB_ORGANIZED_ROOT) / behavior / "train" / "labels" / f"{clip}.xml", behavior)
    return (min(e["start"] for e in events), max(e["end"] for e in events)) if events else None


def build_index() -> list[dict]:
    """창 목록: {clip, fold, behavior, source, start, label, cls}. label: 1=그 행동 양성, 0=정상, -1=제외(애매). cls: 0 정상, 1 전도, 2 유기, 3 절도(6클래스면 4 파손, 5 방화).
    SOURCE에 없는 행동의 영상(4클래스 모드의 파손·방화, 6클래스 모드의 반품)은 건너뛴다."""
    rows = []
    split = list(csv.DictReader(open(SPLIT_PATH, encoding="utf-8")))
    sites = sorted({r["site"] for r in split})
    for r in split:
        p = FULL / f"{r['clip']}.npz"
        # 격리는 전도 헤드 기준(q_fall)을 모든 행동에 같이 적용한다(공식 validation과 날짜·장면이 겹치는 클립은 쓰지 않는다)
        if r["behavior"] not in SOURCE or r["official_split"] != "train" or r["q_fall"] == "True" or not p.is_file() or not (FRAMES / f"{r['clip']}.pkl").is_file():
            continue
        z = np.load(p)
        starts = z["start"].tolist()
        if r["behavior"] == "전도":
            labels = z["label"].tolist()
        elif r["behavior"] in CLASS_OF:
            labels = event_window_labels(r["behavior"], r["clip"], starts)
        else:
            labels = [0] * len(starts)
        span = event_span(r["behavior"], r["clip"]) if (PRE_EVENT_MARGIN or POST_EVENT_MARGIN) and r["behavior"] in CLASS_OF else None
        for s0, y in zip(starts, labels):
            rows.append({"clip": r["clip"], "fold": sites.index(r["site"]) if SITE_FOLDS else int(r["fold"]), "behavior": r["behavior"], "source": SOURCE[r["behavior"]], "start": s0, "label": int(y),
                         "cls": CLASS_OF.get(r["behavior"], 0) if y == 1 else 0,
                         "pre_event": span is not None and 0 <= span[0] - (s0 + WIN) < PRE_EVENT_MARGIN,     # 사건 시작 앞 PRE_EVENT_MARGIN 표본 안인 창
                         "post_event": span is not None and 0 <= s0 - span[1] < POST_EVENT_MARGIN})            # 사건 끝 뒤 POST_EVENT_MARGIN 표본 안인 창
    return rows


def select_clips(rows: list[dict], seed: int = 0) -> set[str]:
    """행동마다 양성 창이 fold당 TARGET_POS/5개가 될 때까지 클립을 시드 고정 순서로 뽑는다(fold 소속은 그대로라 누수 없음)."""
    pos: dict[tuple[str, int, str], int] = defaultdict(int)
    for r in rows:
        if r["behavior"] in CLASS_OF and r["label"] == 1:
            pos[(r["behavior"], r["fold"], r["clip"])] += 1
    rng, chosen = random.Random(seed), set()
    for behavior in CLASS_OF:
        for fold in sorted({f for b, f, _ in pos if b == behavior}):
            clips = sorted(c for b, f, c in pos if b == behavior and f == fold)
            rng.shuffle(clips)
            total = 0
            for c in clips:
                if total >= TARGET_POS // 5:
                    break
                chosen.add(c); total += pos[(behavior, fold, c)]
    return chosen


class Windows(Dataset):
    def __init__(self, items: list[dict], train: bool):
        self.items, self.train, self._cache = items, train, {}

    def __len__(self) -> int:
        return len(self.items)

    def _jpegs(self, clip: str):
        if clip not in self._cache:
            if len(self._cache) > 48:
                self._cache.pop(next(iter(self._cache)))
            data = pickle.loads((FRAMES / f"{clip}.pkl").read_bytes())
            self._cache[clip] = data["jpeg"] if INPUT == "full" else data
        return self._cache[clip]

    def __getitem__(self, i: int):
        it = self.items[i]
        data = self._jpegs(it["clip"])
        jp = data[it["start"]:it["start"] + WIN] if INPUT == "full" else data[it["start"]]
        frames = np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in jp])  # BGR
        frames = frames[np.round(np.linspace(0, len(frames) - 1, NUM_FRAMES)).astype(int)][..., ::-1].astype(np.float32) / 255.0
        if self.train:
            frames = self._augment(frames)
        x = ((frames - MEAN) / STD).transpose(3, 0, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(x)), it["cls"]

    @staticmethod
    def _augment(frames: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            frames = frames[:, :, ::-1]
        scale = random.uniform(0.85, 1.0)
        h = w = int(SIZE * scale)
        y0, x0 = random.randint(0, SIZE - h), random.randint(0, SIZE - w)
        frames = np.stack([cv2.resize(f[y0:y0 + h, x0:x0 + w], (SIZE, SIZE)) for f in frames])
        b, s = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
        if INPUT != "full" and random.random() < 0.2:  # 크롭 모드만: 색 단서(매장 색감)를 덜 쓰게 일부는 흑백으로(A안 fold와 학습 조건을 맞추려고 full에는 적용 안 함)
            s = 0.0
        gray = frames.mean(axis=3, keepdims=True)
        return np.clip((gray + (frames * b - gray) * s), 0.0, 1.0)


def make_model():
    from torchvision.models.video import S3D_Weights, s3d

    model = s3d(weights=S3D_Weights.DEFAULT)
    model.classifier[1] = nn.Conv3d(model.classifier[1].in_channels, len(LABEL_MAP), kernel_size=1, stride=1)
    return model


def audited_windows() -> set[tuple[str, int]]:
    """검수에서 통과한 (영상, 시작 표본) 창. 검수 파일이 하나도 없으면 임의로 비워 두지 않고 예외를 던진다."""
    files = sorted((APP_ROOT / "output").glob("normal_audit_claude_*.csv"))
    files = [f for f in files if f.stem != "normal_audit_claude"]
    if not files:
        raise FileNotFoundError("AUDIT_NORMALS=1인데 output/normal_audit_claude_<장소>.csv가 없다")
    return {(r["clip"], int(r["start"])) for f in files for r in csv.DictReader(open(f, encoding="utf-8")) if r["verdict"] == "pass"}


def _audited_epoch(pos: list[dict], own: dict[str, list[dict]], normal: list[dict], rng: random.Random) -> list[dict]:
    """검수 정상 에폭: 장소마다 정상 수 = 그 장소의 양성 수. 정상 후보 = 검수 통과 창 + 매장이동 창(그 장소). 모자라면 같은 창을 반복한다."""
    site = {r["clip"]: r["site"] for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8"))}
    ok = audited_windows()
    cand = [r for p in own.values() for r in p if (r["clip"], r["start"]) in ok] + normal
    by_site: dict[str, list[dict]] = {}
    for r in cand:
        by_site.setdefault(site[r["clip"]], []).append(r)
    need: dict[str, int] = {}
    for r in pos:
        need[site[r["clip"]]] = need.get(site[r["clip"]], 0) + 1
    neg: list[dict] = []
    for s, n in need.items():
        pool = by_site.get(s, [])
        if not pool:   # 이 fold의 학습에 그 장소의 정상이 없으면(매장이동이 평가 fold에 몰린 경우) 전체 후보에서 뽑고 알린다
            print(f"경고: 장소 {s}에 정상 후보가 없어 전체 후보에서 {n}개를 뽑는다", flush=True)
            pool = cand
        neg += rng.sample(pool, min(n, len(pool))) + (rng.choices(pool, k=n - len(pool)) if n > len(pool) else [])
    items = pos + neg
    rng.shuffle(items)
    return items


def epoch_items(rows: list[dict], fold: int, selected: set[str], rng: random.Random) -> list[dict]:
    """양성: 선택된 클립의 모든 행동 양성 창. 정상 클래스: 양성 총수만큼, 절반은 행동 영상들의 사건 밖 구간(행동당 같은 수), 절반은 정상 풀."""
    train = [r for r in rows if r["fold"] != fold]
    pos = [r for r in train if r["label"] == 1 and r["clip"] in selected]
    own = {b: [r for r in train if r["behavior"] == b and r["label"] == 0 and r["clip"] in selected and not r.get("pre_event") and not r.get("post_event")] for b in CLASS_OF}   # 사건 직전·직후 창은 정상으로 쓰지 않는다(PRE/POST_EVENT_MARGIN)
    if AUDIT_NORMALS:
        own_all = {b: [r for r in train if r["behavior"] == b and r["label"] == 0] for b in CLASS_OF}   # 검수한 창은 양성 선택 영상이 아니어도 쓴다
        return _audited_epoch(pos, own_all, [r for r in train if r["source"] == "normal"], rng)
    normal = [r for r in train if r["source"] == "normal"]
    half, per = len(pos) // 2, len(pos) // (2 * len(CLASS_OF))   # 행동 3개면 1/6, 5개면 1/10씩
    neg = [x for p in own.values() for x in rng.sample(p, min(per, len(p)))] + rng.sample(normal, min(half, len(normal)))
    if len(neg) < len(pos):   # 정상 풀이 작아 모자란 만큼(반품을 뺀 6클래스)은 행동 영상의 사건 밖 구간에서 채운다. 이미 뽑은 창은 다시 뽑지 않는다
        taken = {(x["clip"], x["start"]) for x in neg}
        rest = [x for p in own.values() for x in p if (x["clip"], x["start"]) not in taken]
        neg += rng.sample(rest, min(len(pos) - len(neg), len(rest)))
    items = pos + neg
    rng.shuffle(items)
    return items


@torch.no_grad()
def score_all(model, items: list[dict], device: str, workers: int) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in DataLoader(Windows(items, False), batch_size=16, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x.to(device)).float()
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def run_fold(fold: int, args) -> None:
    rows = build_index()
    selected = select_clips(rows)
    device = "cuda"
    model = make_model().to(device)
    backbone = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": model.classifier.parameters(), "lr": args.lr * 10}], weight_decay=1e-4)
    rng, scaler = random.Random(fold), torch.amp.GradScaler()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for epoch in range(args.epochs):
        items = epoch_items(rows, fold, selected, rng) if fold >= 0 else epoch_items(rows, -1, selected, rng)
        model.train()
        total = 0.0
        counts = np.bincount([i["cls"] for i in items], minlength=len(LABEL_MAP)).astype(np.float32)
        weights = torch.tensor(counts.sum() / (len(LABEL_MAP) * np.maximum(counts, 1)), device=device)  # 클래스 균형(정상 클래스가 더 많다)
        for step, (x, y) in enumerate(DataLoader(Windows(items, True), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(x.to(device)).float(), y.to(device), weight=weights)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item()
            if step % 50 == 0:
                print(f"fold {fold} epoch {epoch + 1}/{args.epochs} step {step} loss {loss.item():.4f}", flush=True)
        sched.step()
        print(f"fold {fold} epoch {epoch + 1} 평균 loss {total / max(step + 1, 1):.4f} (창 {len(items)}개)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "label_map": LABEL_MAP, "arch": "s3d", "size": SIZE, "num_frames": NUM_FRAMES,
                "epoch": args.epochs, "val_acc": None}, OUT / f"s3d_fold{fold}.pt")
    if fold >= 0:  # 보류 fold의 모든 창을 채점
        held = [r for r in rows if r["fold"] == fold]
        scores = score_all(model, held, device, args.workers)  # (창, 클래스 수) 로짓: 정상, 전도, 유기, 절도(6클래스면 + 파손, 방화)
        np.savez_compressed(OUT / f"oof_fold{fold}.npz", clip=np.array([r["clip"] for r in held]), start=np.array([r["start"] for r in held]),
                            label=np.array([r["label"] for r in held]), cls=np.array([r["cls"] for r in held]), behavior=np.array([r["behavior"] for r in held]), logits=scores)
        print(f"fold {fold} 보류 창 {len(held)}개 채점 완료", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="0~4: 그 fold를 보류하고 학습·채점 / -1: 모든 fold로 최종 학습(동결 후)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="백본 학습률(분류 층은 10배)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    torch.manual_seed(0), random.seed(0), np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(float(os.getenv("GPU_MEM_FRACTION", "0.4")))  # 다른 GPU 작업(탐지기 학습·시연)과 나눠 쓰기 위한 VRAM 상한
    run_fold(args.fold, args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
