#!/usr/bin/env python3
"""
extract_event_frames.py로 만든 클립 폴더 구조를 학습시키는 스크립트.

입력 구조 (extract_event_frames.py --extract_normals 결과):
    dataset_dir/
        manifest.csv                     <- label 컬럼(fall/normal) 포함
        {영상명}__fall/     frame_000001.jpg ... frame_0000NN.jpg
        {영상명}__normal_1/ frame_000001.jpg ... frame_0000NN.jpg

핵심 설계:
- source_video 단위로 train/val을 나눈다 (같은 원본 영상에서 나온 fall/normal
  클립이 train과 val로 쪼개지면 배경/인물 누수가 생기기 때문)
- 데이터가 매우 적으므로(수십 클립) Kinetics 사전학습 백본은 필수.
  기본값은 백본을 얼리고 classifier head만 학습(freeze)한다.

사용법:
    # 1단계: head만 학습 (권장 시작점)
    python3 train_i3d.py --data ~/aihub_fall_dataset --epochs 30 --freeze_backbone

    # 2단계: 전체 미세조정 (1단계 후, 낮은 lr로)
    python3 train_i3d.py --data ~/aihub_fall_dataset --epochs 20 --lr 1e-4 \
        --resume runs/best.pt
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

# Kinetics-400 정규화 상수 (torchvision video model 기준)
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


# ---------------------------------------------------------------- 데이터 준비

def read_manifest(data_dir: Path):
    """manifest.csv를 읽어 클립 레코드 리스트를 만든다."""
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.csv 없음: {manifest}")

    records = []
    with open(manifest, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clip_dir = data_dir / row["clip_folder"]
            if not clip_dir.is_dir():
                print(f"[경고] 폴더 없음, 건너뜀: {row['clip_folder']}")
                continue
            frames = sorted(clip_dir.glob("frame_*.jpg"))
            if not frames:
                print(f"[경고] 프레임 없음, 건너뜀: {row['clip_folder']}")
                continue
            records.append({
                "clip_folder": row["clip_folder"],
                "clip_dir": clip_dir,
                "frames": frames,
                "label": row.get("label") or row.get("event_type"),
                "source_video": row.get("source_video", row["clip_folder"]),
            })
    return records


def split_by_source_video(records, val_ratio=0.25, seed=42):
    """같은 원본 영상에서 나온 클립은 통째로 같은 split에 들어가게 나눈다.
    (fall 클립과 그 짝인 normal 클립이 train/val로 갈리면 누수)"""
    by_video = defaultdict(list)
    for r in records:
        by_video[r["source_video"]].append(r)

    videos = sorted(by_video)
    rng = random.Random(seed)
    rng.shuffle(videos)

    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = set(videos[:n_val])

    train, val = [], []
    for v in videos:
        (val if v in val_videos else train).extend(by_video[v])
    return train, val, sorted(val_videos)


def build_label_map(records):
    labels = sorted({r["label"] for r in records})
    return {name: i for i, name in enumerate(labels)}


# ---------------------------------------------------------------- 프레임 로딩

def load_clip(frames, size=224, train_mode=False, rng=None):
    """클립 폴더의 프레임들을 (C, T, H, W) float32 배열로 읽는다.

    augmentation은 클립 전체에 '동일하게' 적용해야 한다.
    프레임마다 다르게 뒤집거나 크롭하면 시간적 연속성이 깨진다.
    """
    import cv2

    rng = rng or random.Random()
    do_flip = train_mode and rng.random() < 0.5
    # 색상 지터: 외형(옷 색/조명) 의존을 낮추기 위해 클립 단위로 한 번만 결정
    if train_mode:
        brightness = rng.uniform(0.8, 1.2)
        saturation = rng.uniform(0.8, 1.2)
    else:
        brightness = saturation = 1.0

    arr = []
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            raise RuntimeError(f"이미지 읽기 실패: {fp}")
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if do_flip:
            img = img[:, ::-1, :]

        img = img.astype(np.float32) / 255.0

        if train_mode and (brightness != 1.0 or saturation != 1.0):
            img = img * brightness
            gray = img.mean(axis=2, keepdims=True)
            img = gray + (img - gray) * saturation
            img = np.clip(img, 0.0, 1.0)

        img = (img - MEAN) / STD
        arr.append(img)

    clip = np.stack(arr, axis=0)          # (T, H, W, C)
    clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
    return np.ascontiguousarray(clip, dtype=np.float32)


# ---------------------------------------------------------------- 학습 본체

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="학습용 클립 폴더 (extract_event_frames.py 출력)")
    ap.add_argument("--val_data", default=None,
                     help="검증용 클립 폴더를 따로 지정. 생략하면 --data를 원본영상 단위로 자동 분할")
    ap.add_argument("--out", default="./runs", help="체크포인트 저장 폴더")
    ap.add_argument("--arch", default="r3d_18",
                     choices=["r3d_18", "mc3_18", "r2plus1d_18", "s3d"],
                     help="백본. s3d가 I3D와 가장 가까운 구조 (torchvision 제공)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224, help="프레임 리사이즈 크기")
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_backbone", action="store_true",
                     help="백본을 얼리고 classifier head만 학습 (데이터 적을 때 권장)")
    ap.add_argument("--resume", default=None, help="이어서 학습할 체크포인트 경로")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest(data_dir)
    if not records:
        raise SystemExit("클립을 하나도 못 찾았습니다. --data 경로를 확인하세요.")

    if args.val_data:
        # val 폴더를 직접 지정한 경우: 자동 분할하지 않는다.
        val_dir = Path(args.val_data).expanduser()
        val_recs = read_manifest(val_dir)
        if not val_recs:
            raise SystemExit(f"검증 클립을 못 찾았습니다: {val_dir}")
        train_recs = records

        # 안전장치: train과 val에 같은 원본 영상이 섞여있으면 누수다.
        train_srcs = {r["source_video"] for r in train_recs}
        val_srcs = {r["source_video"] for r in val_recs}
        overlap = train_srcs & val_srcs
        if overlap:
            print(f"\n[경고] train과 val에 같은 원본 영상이 {len(overlap)}개 겹칩니다:")
            for s in sorted(overlap)[:5]:
                print(f"   - {s}")
            if len(overlap) > 5:
                print(f"   ... 외 {len(overlap)-5}개")
            print("  -> val 점수가 부풀려집니다. 폴더 구성을 확인하세요.\n")

        # 라벨 맵은 양쪽을 합쳐서 만들어야 클래스 인덱스가 어긋나지 않는다.
        label_map = build_label_map(records + val_recs)
    else:
        label_map = build_label_map(records)
        train_recs, val_recs, val_videos = split_by_source_video(
            records, val_ratio=args.val_ratio, seed=args.seed)
        print(f"(자동 분할) val 원본영상 {len(val_videos)}개")

    print(f"클래스 {label_map}")

    counts = defaultdict(int)
    for r in train_recs:
        counts[r["label"]] += 1
    print("train 클래스별 개수:", dict(counts))
    val_counts = defaultdict(int)
    for r in val_recs:
        val_counts[r["label"]] += 1
    print("val 클래스별 개수:", dict(val_counts))

    print(f"train {len(train_recs)}클립 / val {len(val_recs)}클립")
    if not val_recs:
        raise SystemExit("val이 비었습니다.")

    # 클래스 가중치는 train 분포 기준으로 계산
    for name in label_map:
        counts.setdefault(name, 0)

    class ClipDataset(Dataset):
        def __init__(self, recs, train_mode):
            self.recs = recs
            self.train_mode = train_mode

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            r = self.recs[idx]
            rng = random.Random(hash((r["clip_folder"], idx)) & 0xFFFFFFFF)
            clip = load_clip(r["frames"], size=args.size,
                              train_mode=self.train_mode, rng=rng)
            return torch.from_numpy(clip), label_map[r["label"]]

    train_loader = DataLoader(ClipDataset(train_recs, True), batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(ClipDataset(val_recs, False), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # ---- 모델 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    weights_arg = "KINETICS400_V1"  # Kinetics-400 사전학습 (데이터 적을 때 필수)
    builder = getattr(torchvision.models.video, args.arch)
    model = builder(weights=weights_arg)

    # classifier head를 클래스 수에 맞게 교체
    n_classes = len(label_map)
    if args.arch == "s3d":
        in_ch = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv3d(in_ch, n_classes, kernel_size=1, stride=1)
    else:
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    if args.freeze_backbone:
        head_names = ("classifier", "fc")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(head_names)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"백본 동결. 학습 파라미터 {n_train:,}개")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"체크포인트 로드: {args.resume}")

    model.to(device)

    # 클래스 불균형 보정 (train 분포 기준, 0으로 나누기 방지)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (n_classes * counts[name]) if counts[name] > 0 else 1.0
         for name in label_map],
        dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)

        model.eval()
        va_loss = va_correct = va_total = 0
        per_class = defaultdict(lambda: [0, 0])  # label -> [correct, total]
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                loss = criterion(logits, labels)
                preds = logits.argmax(1)

                va_loss += loss.item() * labels.size(0)
                va_correct += (preds == labels).sum().item()
                va_total += labels.size(0)
                for p, t in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    per_class[t][1] += 1
                    if p == t:
                        per_class[t][0] += 1

        scheduler.step()
        tr_acc = tr_correct / max(tr_total, 1)
        va_acc = va_correct / max(va_total, 1)
        inv_map = {v: k for k, v in label_map.items()}
        cls_str = " ".join(
            f"{inv_map[c]}={per_class[c][0]}/{per_class[c][1]}" for c in sorted(per_class))
        print(f"[{epoch:3d}/{args.epochs}] "
              f"train loss {tr_loss/max(tr_total,1):.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss/max(va_total,1):.4f} acc {va_acc:.3f} | {cls_str}")

        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "label_map": label_map,
                "arch": args.arch,
                "size": args.size,
                "num_frames": len(train_recs[0]["frames"]),
                "epoch": epoch,
                "val_acc": va_acc,
            }, out_dir / "best.pt")

    (out_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최고 val 정확도: {best_acc:.3f}")
    print(f"저장: {(out_dir / 'best.pt').resolve()}")
    print("\n[주의] 클립 수가 적으면 val 정확도는 변동이 큽니다. "
          "단일 숫자보다 여러 seed로 반복해 평균을 보세요.")


if __name__ == "__main__":
    main()
