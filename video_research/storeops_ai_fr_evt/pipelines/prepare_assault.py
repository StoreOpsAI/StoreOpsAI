"""Build session-safe assault windows without reading official validation video frames.

The assault XML uses <event> and per-person <action> tags, unlike the
abandonment annotations. Only contact actions are positive; normal windows
stay outside the entire event with a margin.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from pipelines.aihub_abandonment import APP_ROOT, WORKSPACE_ROOT


CONTACT = {"punching", "kicking", "pushing", "pulling", "throwing"}
WINDOW_SEC = 4
MARGIN_SEC = 2


def seconds(value: str) -> float:
    hours, minutes, secs = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(secs)


def windows(xml: Path, video: Path, split: str) -> list[dict]:
    root = ET.parse(xml).getroot()
    fps = float(root.findtext("header/fps"))
    total = int(root.findtext("header/frames"))
    event = root.find("event")
    if fps <= 0 or total <= 0 or event is None or event.findtext("eventname") != "assault":
        raise ValueError(f"Invalid assault annotation: {xml}")
    start = round(seconds(event.findtext("starttime")) * fps)
    end = round((seconds(event.findtext("starttime")) + seconds(event.findtext("duration"))) * fps)
    if not 0 <= start < end <= total:
        raise ValueError(f"Invalid event interval: {xml}")

    length = round(WINDOW_SEC * fps)
    margin = round(MARGIN_SEC * fps)
    group = video.stem.split("_cam", 1)[0]
    if group == video.stem:
        raise ValueError(f"Missing camera/session key: {video}")
    base = {"split": split, "group": group, "video": str(video.resolve()), "fps": fps}
    rows = []
    positive = set()
    for action in root.findall(".//object/action"):
        if action.findtext("actionname") not in CONTACT:
            continue
        for span in action.findall("frame"):
            first, last = int(span.findtext("start")), int(span.findtext("end"))
            if first > last or first < start or last > end:
                raise ValueError(f"Contact action outside event: {xml}")
            clip_start = max(start, min((first + last - length) // 2, end - length))
            if clip_start >= 0 and clip_start + length <= total:
                positive.add(clip_start)
    for frame in sorted(positive):
        rows.append({**base, "label": "assault", "start_frame": frame, "end_frame": frame + length - 1})

    # Keep normal clips clear of transitions and of every assault action.
    available = [(0, max(0, start - margin)), (min(total, end + margin), total)]
    for lo, hi in available:
        if hi - lo >= length:
            frame = lo + (hi - lo - length) // 2
            rows.append({**base, "label": "normal", "start_frame": frame, "end_frame": frame + length - 1})
    return rows


def build(root: Path) -> list[dict]:
    rows = []
    groups = {}
    for folder, split in (("Training", "train"), ("Validation", "test")):
        labels, videos = root / folder / "labels", root / folder / "videos"
        for xml in sorted(labels.glob("*.xml")):
            video = videos / f"{xml.stem}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            samples = windows(xml, video, split)
            if not any(row["label"] == "assault" for row in samples):
                raise ValueError(f"No assault contact windows: {xml}")
            for row in samples:
                previous = groups.setdefault(row["group"], split)
                if previous != split:
                    raise ValueError(f"Train/test session overlap: {row['group']}")
            rows.extend(samples)
    if not rows:
        raise ValueError(f"No assault annotations under {root}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=WORKSPACE_ROOT / "폭행")
    parser.add_argument("--output", type=Path, default=APP_ROOT / "output" / "assault_windows.csv")
    args = parser.parse_args()
    rows = build(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "test"):
        subset = [row for row in rows if row["split"] == split]
        print(split, "sessions", len({row["group"] for row in subset}),
              "assault", sum(row["label"] == "assault" for row in subset),
              "normal", sum(row["label"] == "normal" for row in subset))
    print(args.output)


if __name__ == "__main__":
    main()
