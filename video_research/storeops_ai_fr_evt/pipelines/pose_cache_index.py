"""전도·정상 포즈 캐시(output/lying_pose)에 스펙 메타데이터를 붙인 색인과, 라벨(xml) 관절 대비 YOLO-pose 관절 오차를 만든다.

- 색인(output/pose_cache/index.csv): 영상 ID·그룹(촬영 사이트_날짜)·fold·split·fps·추출기 해시·git 커밋. 영상을 지운 뒤에도 분할·겹침 검사가 되도록.
- 관절 오차(output/pose_cache/joint_error.json): 라벨 사람마다 같은 프레임의 YOLO 사람 중 관절 평균 거리가 가장 가까운 한 명과 짝지어(정답에 유리한 짝짓기) 몸통 길이(목-골반)로 나눈 관절별 오차를 낸다.
  임계값은 만들지 않는다. 측정값만 저장한다. 라벨에 없는 코·눈·귀는 비교하지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from config.config import POSE_IMGSZ, POSE_WEIGHTS
from pipelines.aihub_abandonment import APP_ROOT
from pipelines.lying_manifest import MANIFEST_PATH
from pipelines.lying_pose_extract import OUTPUT_DIR, PERSON_CONF
from pipelines.normal_manifest import NORMAL_MANIFEST_PATH

OUT = APP_ROOT / "output" / "pose_cache"
SAMPLE_FPS = 3.0  # 클립은 3fps로 저장돼 있다(180프레임 = 60초)
# xml 관절 이름 -> COCO 번호. 손=손목, 발=발목. 라벨 이름에 공백이 두 칸인 곳이 있어 공백을 정리해 비교한다.
JOINTS = {"left shoulder": 5, "right shoulder": 6, "left elbow": 7, "right elbow": 8, "left hand": 9, "right hand": 10,
          "left hip": 11, "right hip": 12, "left knee": 13, "right knee": 14, "left foot": 15, "right foot": 16}
COCO_NAMES = {v: k for k, v in JOINTS.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=APP_ROOT).stdout.strip()


def _clips() -> list[dict]:
    lying = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    normal = json.loads(NORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = [dict(c, kind="fall", split="train") for c in lying["train"]]
    rows += [dict(c, kind="fall", split="validation") for c in lying["validation"]]
    rows += [dict(c, kind="normal", split="train") for c in normal["train"]]
    rows += [dict(c, kind="normal", split="validation") for c in normal["validation"]]
    return rows


def build_index() -> list[dict]:
    meta = {"extractor": POSE_WEIGHTS, "extractor_sha256": _sha256(APP_ROOT / POSE_WEIGHTS), "git_commit": _git_commit(),
            "imgsz": POSE_IMGSZ, "candidate_person_conf": PERSON_CONF, "fps": SAMPLE_FPS}
    rows = []
    for c in _clips():
        cache = OUTPUT_DIR / f"{c['clip']}.npz"
        if not cache.is_file():
            continue
        rows.append({"video_id": c["clip"], "kind": c["kind"], "split": c["split"], "group_id": f"{c['site']}_{c['date']}",
                     "scene": c["scene"], "camera": c["camera"], "fold": c.get("fold", ""), "quarantine": c.get("quarantine", False),
                     "first_pass": c.get("first_pass", False), "n_frames": int(np.load(cache)["n_frames"]), "cache": str(cache), **meta})
    return rows


def parse_gt_joints(xml_path: Path) -> dict[int, dict[int, np.ndarray]]:
    """{프레임: {사람 ID: 17x2 (라벨에 없는 관절은 NaN)}}"""
    people: dict[int, dict[int, np.ndarray]] = {}
    for track in ET.parse(xml_path).findall(".//track"):
        number = JOINTS.get(" ".join(track.get("label", "").lower().split()))
        if number is None:
            continue
        for p in track.findall("points"):
            if p.get("outside") == "1":
                continue
            pid = int(next((a.text for a in p.findall("attribute") if a.get("name") == "ID"), "0"))
            x, y = map(float, p.get("points").split(";")[0].split(","))
            frame = int(p.get("frame"))
            people.setdefault(frame, {}).setdefault(pid, np.full((17, 2), np.nan))[number] = (x, y)
    return people


def _torso(xml_path: Path) -> dict[int, dict[int, float]]:
    """{프레임: {사람 ID: 목-골반 길이}}"""
    out: dict[int, dict[int, float]] = {}
    pts: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for track in ET.parse(xml_path).findall(".//track"):
        label = " ".join(track.get("label", "").lower().split())
        if label not in ("neck base", "pelvis"):
            continue
        for p in track.findall("points"):
            pid = int(next((a.text for a in p.findall("attribute") if a.get("name") == "ID"), "0"))
            x, y = map(float, p.get("points").split(";")[0].split(","))
            pts.setdefault((int(p.get("frame")), pid), {})[label] = np.array([x, y])
    for (frame, pid), d in pts.items():
        if len(d) == 2:
            out.setdefault(frame, {})[pid] = float(np.linalg.norm(d["neck base"] - d["pelvis"]))
    return out


def clip_errors(clip: dict) -> list[tuple[int, bool, np.ndarray, float]]:
    """(관절 번호 길이 17의 몸통 정규화 오차, 누움 구간 여부, 관절 신뢰도 17) 목록이 아니라 (frame, lying, err[17], conf 평균) 행."""
    cache = np.load(OUTPUT_DIR / f"{clip['clip']}.npz")
    xml = Path(clip["video"]).parent.parent / "labels" / f"{clip['clip']}.xml"
    gt, torso = parse_gt_joints(xml), _torso(xml)
    rows = []
    for frame, persons in gt.items():
        det = np.where(cache["frame"] == frame)[0]
        if len(det) == 0:
            continue
        for pid, joints in persons.items():
            length = torso.get(frame, {}).get(pid)
            if not length:
                continue
            valid = ~np.isnan(joints[:, 0])
            dists = [np.linalg.norm(cache["kpt"][i][valid, :2] - joints[valid], axis=1).mean() for i in det]
            best = det[int(np.argmin(dists))]
            err = np.full(17, np.nan)
            err[valid] = np.linalg.norm(cache["kpt"][best][valid, :2] - joints[valid], axis=1) / length
            lying = any(f["start"] <= frame <= f["end"] for f in clip["falls"])
            rows.append((frame, lying, err, cache["kpt"][best][:, 2]))
    return rows


def summarize(clips: list[dict]) -> dict:
    groups: dict[str, list] = {"lying": [], "other": []}
    for clip in clips:
        for _, lying, err, conf in clip_errors(clip):
            groups["lying" if lying else "other"].append(np.concatenate([err, conf]))
    result = {}
    for name, rows in groups.items():
        if not rows:  # 라벨 관절은 누운 구간에만 있다(other는 비어 있을 수 있다)
            continue
        arr = np.array(rows)
        result[name] = {"samples": len(rows), "joints": {
            COCO_NAMES[j]: {"median": float(np.nanmedian(arr[:, j])), "p90": float(np.nanpercentile(arr[:, j], 90)),
                            "mean_conf": float(np.nanmean(arr[:, 17 + j]))} for j in COCO_NAMES}}
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_index()
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"색인 {len(rows)}개 -> {OUT / 'index.csv'}")
    # 관절 오차는 공식 validation을 쓰지 않는다(AGENTS.md 5). 라벨이 있는 전도 train 클립만.
    done = {r["video_id"] for r in rows}
    clips = [c for c in _clips() if c["kind"] == "fall" and c["split"] == "train" and not c["quarantine"] and c["clip"] in done]
    report = {"clips": len(clips), "note": "정답 유리한 짝짓기, 몸통 길이 정규화, 임계값 없음", **summarize(clips)}
    (OUT / "joint_error.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in report.keys() & {"lying", "other"}:
        print(name, report[name]["samples"], {k: round(v["median"], 3) for k, v in report[name]["joints"].items()})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
