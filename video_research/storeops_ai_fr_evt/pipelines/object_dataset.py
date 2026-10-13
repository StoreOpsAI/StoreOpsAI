"""유기 라벨의 물건 트랙(object_abandon, 보간된 프레임별 박스)으로 YOLO 물건 탐지 학습 데이터를 만든다.

- 클래스 1개(0 object). 라벨에는 유기 대상 물건만 있고 다른 물건은 배경으로 남는다(알려진 한계).
- 분할은 abandon_manifest의 세션 fold를 따른다: TEST_FOLD는 만들지 않고(평가는 eval_object_recall), VAL_FOLD는 체크포인트 선택용,
  나머지 fold는 train. 공식 validation과 quarantine 클립은 쓰지 않는다. 같은 세션이라 같은 fold에 있다(AGENTS.md 규칙 1·2·6).
- 프레임은 OBJECT_FT_FRAME_STRIDE 간격으로 뽑아 긴 변 OBJECT_FT_IMG_SIDE로 줄여 저장한다.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

from config.config import (OBJECT_FT_DIR, OBJECT_FT_FRAME_STRIDE, OBJECT_FT_IMG_SIDE, OBJECT_FT_TEST_FOLD,
                           OBJECT_FT_TEST_SITE, OBJECT_FT_VAL_FOLD)
from pipelines.abandon_manifest import ABANDON_MANIFEST_PATH


def read_track_boxes(xml_path: Path) -> dict[int, list[list[float]]]:
    """프레임 번호 -> 그 프레임의 물건 박스들(x1,y1,x2,y2). 보이는(outside=0) 박스만."""
    boxes = defaultdict(list)
    for track in ET.parse(xml_path).getroot().iter("track"):
        if track.get("label") != "object_abandon":
            continue
        for box in track.iter("box"):
            if box.get("outside") == "0":
                boxes[int(box.get("frame"))].append([float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")])
    return boxes


def yolo_line(box, width, height) -> str:
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_clip(clip: dict, role: str, root: Path) -> int:
    xml_path = Path(clip["video"]).parent.parent / "labels" / f"{clip['clip']}.xml"
    boxes = read_track_boxes(xml_path)
    frames = sorted(f for f in boxes if f % OBJECT_FT_FRAME_STRIDE == 0)
    if not frames:
        return 0
    img_dir, lbl_dir = root / "images" / role, root / "labels" / role
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(clip["video"])
    done = 0
    for frame_no in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"프레임을 읽지 못함: {clip['clip']} {frame_no}")
        h, w = frame.shape[:2]
        scale = OBJECT_FT_IMG_SIDE / max(h, w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
        name = f"{clip['clip']}_{frame_no:03d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / f"{name}.txt").write_text("\n".join(yolo_line(b, w, h) for b in boxes[frame_no]), encoding="utf-8")
        done += 1
    cap.release()
    return done


def is_test(clip: dict) -> bool:
    """시험 영상: 사이트가 지정되면 그 매장 전부, 아니면 TEST_FOLD."""
    return clip["site"] == OBJECT_FT_TEST_SITE if OBJECT_FT_TEST_SITE else clip["fold"] == OBJECT_FT_TEST_FOLD


def split_clips(manifest: dict) -> dict[str, list[dict]]:
    roles = {"train": [], "val": []}
    for clip in manifest["train"]:
        if clip["quarantine"] or is_test(clip):
            continue
        roles["val" if clip["fold"] == OBJECT_FT_VAL_FOLD else "train"].append(clip)
    return roles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="영상 수와 세션만 세고 쓰지 않는다")
    args = parser.parse_args()
    manifest = json.loads(Path(ABANDON_MANIFEST_PATH).read_text(encoding="utf-8"))
    roles = split_clips(manifest)
    root = Path(OBJECT_FT_DIR)
    sessions = {r: {c["session"] for c in clips} for r, clips in roles.items()}
    assert not sessions["train"] & sessions["val"], "train과 val이 세션을 공유"
    test_sessions = {c["session"] for c in manifest["train"] if is_test(c)}
    assert not test_sessions & (sessions["train"] | sessions["val"]), "시험 fold가 세션을 공유"
    for role, clips in roles.items():
        print(f"{role}: 영상 {len(clips)}개, 세션 {len(sessions[role])}개")
    if args.dry_run:
        return
    counts = {role: sum(export_clip(c, role, root) for c in clips) for role, clips in roles.items()}
    (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n",
                                    encoding="utf-8")
    (root / "split.json").write_text(json.dumps({"test_fold": OBJECT_FT_TEST_FOLD, "test_site": OBJECT_FT_TEST_SITE, "val_fold": OBJECT_FT_VAL_FOLD,
                                                 "images": counts}, indent=1), encoding="utf-8")
    print("이미지 수:", counts)


if __name__ == "__main__":
    main()
