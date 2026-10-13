"""얼린 사전학습 영상 백본 비교: S3D / VideoMAE-B / X-CLIP (유기 vs 정상).

- 공식 validation(test) 샘플은 특징도 뽑지 않는다.
- 백본은 학습하지 않고 클립 특징을 한 번만 뽑아 캐시한다.
- 선형 끝단은 cv_abandonment와 같은 세션 단위 fold로 학습·평가한다(threshold 0.5, 하이퍼파라미터 고정).
- X-CLIP은 추가로 학습 없이 문장 비교(zero-shot) 점수도 낸다. 문장은 미리 정해 두고 결과를 보고 고치지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config.config import VIDEOMAE_MODEL, XCLIP_MODEL
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.cv_abandonment import _summary, assign_folds
from pipelines.train_abandonment import MEAN, STD, _metrics

ABANDON_PROMPTS = ["a person putting down a bag on the floor and walking away",
                   "a person leaving an object behind"]
NORMAL_PROMPTS = ["a person walking in a store", "a person standing in a store",
                  "a person looking at products on a shelf"]


def _load_frames(row):
    return np.load(row["path"])["frames"]  # (16, 224, 224, 3) RGB uint8


@torch.no_grad()
def extract_s3d(rows, device):
    from torchvision.models.video import S3D_Weights, s3d

    model = s3d(weights=S3D_Weights.DEFAULT).features.to(device).eval()
    features = []
    for row in rows:
        clip = torch.from_numpy(_load_frames(row)).permute(3, 0, 1, 2).float() / 255
        clip = ((clip - MEAN) / STD)[None].to(device)
        features.append(model(clip).mean(dim=(2, 3, 4))[0].cpu().numpy())
    return np.stack(features)


def load_videomae(device):
    from transformers import AutoImageProcessor, VideoMAEForVideoClassification

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    processor = AutoImageProcessor.from_pretrained(VIDEOMAE_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL)
    # transformers 5는 체크포인트의 q_bias/v_bias를 읽지 않고 query/key/value.bias를 새로 만든다.
    # 원본 VideoMAE 구조(key bias 없음)대로 직접 채운다.
    state = load_file(hf_hub_download(VIDEOMAE_MODEL, "model.safetensors"))
    with torch.no_grad():
        for index, layer in enumerate(model.videomae.encoder.layer):
            attention, prefix = layer.attention.attention, f"videomae.encoder.layer.{index}.attention.attention."
            attention.query.bias.copy_(state[prefix + "q_bias"])
            attention.key.bias.zero_()
            attention.value.bias.copy_(state[prefix + "v_bias"])
    return processor, model.to(device).eval()


@torch.no_grad()
def videomae_embed(processor, model, clips):
    """clips: (16, 224, 224, 3) RGB uint8 배열 목록 → 분류층 바로 앞 특징 (N, 768)."""
    inputs = processor([list(clip) for clip in clips], return_tensors="pt").to(model.device)
    hidden = model.videomae(**inputs).last_hidden_state.mean(1)
    return model.fc_norm(hidden).cpu().numpy()


def extract_videomae(rows, device):
    processor, model = load_videomae(device)
    return np.concatenate([videomae_embed(processor, model, [_load_frames(row)]) for row in rows])


def load_xclip(device):
    from transformers import AutoProcessor, XCLIPModel

    return AutoProcessor.from_pretrained(XCLIP_MODEL), XCLIPModel.from_pretrained(XCLIP_MODEL).to(device).eval()


@torch.no_grad()
def xclip_embed(processor, model, clips):
    """clips: (16, 224, 224, 3) RGB uint8 배열 목록 → 정규화한 영상 임베딩 (N, 512). 8프레임만 쓴다."""
    inputs = processor(videos=[list(clip[::2]) for clip in clips], return_tensors="pt").to(model.device)
    embeds = model.get_video_features(**inputs)
    embeds = getattr(embeds, "pooler_output", embeds)
    return (embeds / embeds.norm(dim=-1, keepdim=True)).cpu().numpy()


@torch.no_grad()
def extract_xclip(rows, device):
    """(영상 임베딩, zero-shot 유기 확률)을 돌려준다. X-CLIP base는 8프레임을 받는다."""
    processor, model = load_xclip(device)
    prompts = ABANDON_PROMPTS + NORMAL_PROMPTS
    features, zero_shot = [], []
    for row in rows:
        inputs = processor(text=prompts, videos=list(_load_frames(row)[::2]), return_tensors="pt",
                           padding=True).to(device)
        output = model(**inputs)
        probabilities = output.logits_per_video.softmax(1)[0]
        zero_shot.append(float(probabilities[:len(ABANDON_PROMPTS)].sum()))
        features.append(output.video_embeds[0].cpu().numpy())
    return np.stack(features), np.array(zero_shot)


def _fit_linear(x_train, y_train, epochs=300):
    """표준화 + 클래스 균형 로지스틱 회귀. 설정은 고정이고 dev로 고르지 않는다."""
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    y = torch.tensor(y_train)
    torch.manual_seed(42)
    model = nn.Linear(x.shape[1], 2)
    counts = torch.bincount(y, minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    @torch.no_grad()
    def predict(values):
        return model(torch.tensor((values - mean) / std, dtype=torch.float32)).softmax(1)[:, 1].numpy()

    return predict


def cross_validate_features(features, labels, fold_ids, folds, sources=None):
    per_fold, out_of_fold = [], np.zeros(len(labels))
    for fold in range(folds):
        dev = fold_ids == fold
        predict = _fit_linear(features[~dev], labels[~dev])
        out_of_fold[dev] = predict(features[dev])
        train_metrics = _metrics(labels[~dev].tolist(), predict(features[~dev]).tolist())
        dev_metrics = _metrics(labels[dev].tolist(), predict(features[dev]).tolist())
        per_fold.append({"fold": fold, "train": train_metrics, "dev": dev_metrics,
                         "gap_f1": train_metrics["f1"] - dev_metrics["f1"]})
    report = _report(per_fold)
    if sources is not None:  # 출처별로 dev에서 유기로 판정된 비율(정상 출처면 오탐률)
        report["dev_positive_rate_by_source"] = {
            source: float((out_of_fold[sources == source] >= 0.5).mean()) for source in sorted(set(sources))}
    return report


def _report(per_fold):
    return {
        "dev_f1_유기": _summary([item["dev"]["f1"] for item in per_fold]),
        "dev_f1_정상": _summary([item["dev"]["per_class"]["정상"]["f1"] for item in per_fold]),
        "gap_f1": _summary([item.get("gap_f1", 0.0) for item in per_fold]),
        "dev_confusion_matrix_total": np.sum([item["dev"]["confusion_matrix"] for item in per_fold], axis=0).tolist(),
        "per_fold": per_fold,
    }


def _cached(path, rows, extract):
    ids = [row["sample_id"] for row in rows]
    if path.is_file():
        data = np.load(path)
        if data["ids"].tolist() == ids:
            return {key: data[key] for key in data.files if key != "ids"}
    values = extract()
    values = values if isinstance(values, dict) else {"features": values}
    np.savez(path, ids=np.array(ids), **values)
    return values


def main():
    parser = argparse.ArgumentParser(description="얼린 영상 백본 비교 (공식 validation 미사용)")
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "output" / "abandonment_training" / "samples.csv")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "output" / "backbone_compare")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(APP_ROOT / "output" / "torch_cache"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] in ("train", "dev")]
    fold_of = assign_folds(rows, args.folds)
    fold_ids = np.array([fold_of[row["group"]] for row in rows])
    labels = np.array([int(row["label"]) for row in rows])
    sources = np.array([row["source"] for row in rows]) if "source" in rows[0] else None
    for fold in range(args.folds):
        dev_groups = {row["group"] for row, f in zip(rows, fold_ids) if f == fold}
        train_groups = {row["group"] for row, f in zip(rows, fold_ids) if f != fold}
        if dev_groups & train_groups:
            raise RuntimeError(f"fold {fold}: train/dev 세션이 겹칩니다")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def xclip():
        features, zero_shot = extract_xclip(rows, device)
        return {"features": features, "zero_shot": zero_shot}

    s3d_data = _cached(args.output_dir / "s3d.npz", rows, lambda: extract_s3d(rows, device))
    videomae_data = _cached(args.output_dir / "videomae.npz", rows, lambda: extract_videomae(rows, device))
    xclip_data = _cached(args.output_dir / "xclip.npz", rows, xclip)

    results = {
        "samples": {"정상": int((labels == 0).sum()), "유기": int((labels == 1).sum())},
        "s3d_linear": cross_validate_features(s3d_data["features"], labels, fold_ids, args.folds, sources),
        "videomae_linear": cross_validate_features(videomae_data["features"], labels, fold_ids, args.folds, sources),
        "xclip_linear": cross_validate_features(xclip_data["features"], labels, fold_ids, args.folds, sources),
        "xclip_zero_shot": _report([
            {"fold": fold, "dev": _metrics(labels[fold_ids == fold].tolist(),
                                           xclip_data["zero_shot"][fold_ids == fold].tolist())}
            for fold in range(args.folds)
        ]),
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in result.items() if key != "per_fold"}
                      if isinstance(result, dict) and "per_fold" in result else result
                      for name, result in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
