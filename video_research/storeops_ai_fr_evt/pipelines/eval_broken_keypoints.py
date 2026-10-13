"""파손 타당성 확인: 라벨 관절(GT)로 파손(던지기)과 반품(상품 되돌리기)의 손목 움직임이 갈리는지 본다. 공식 validation 미사용."""
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from pipelines.aihub_abandonment import WORKSPACE_ROOT as ROOT

PARTS = {"Left hand": "lw", "Right hand": "rw", "Left shoulder": "ls", "Right shoulder": "rs",
         "Neck base": "neck", "Pelvis": "pelvis"}


def people(xml):
    """{ID: {frame: {part: (x, y)}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for track in ET.parse(xml).findall(".//track"):
        part = PARTS.get(track.get("label"))
        if not part:
            continue
        for p in track.findall("points"):
            if p.get("outside") == "1":
                continue
            pid = next((a.text for a in p.findall("attribute") if a.get("name") == "ID"), "0")
            x, y = map(float, p.get("points").split(";")[0].split(","))
            out[pid][int(p.get("frame"))][part] = np.array([x, y])
    return out


def features(xml):
    """사람별 (최대 손목 속도, 최대 팔 뻗음, 손이 어깨 위로 올라간 프레임 비율). 몸통 길이로 나눈다."""
    result = []
    for frames in people(xml).values():
        order = sorted(f for f in frames if all(k in frames[f] for k in PARTS.values()))
        if len(order) < 3:
            continue
        torso = np.median([np.linalg.norm(frames[f]["neck"] - frames[f]["pelvis"]) for f in order]) + 1e-6
        speed, reach, raised = [], [], []
        for a, b in zip(order, order[1:]):
            if b - a != 1:
                continue
            speed.append(max(np.linalg.norm(frames[b][w] - frames[a][w]) for w in ("lw", "rw")) / torso)
        for f in order:
            k = frames[f]
            reach.append(max(np.linalg.norm(k["lw"] - k["ls"]), np.linalg.norm(k["rw"] - k["rs"])) / torso)
            raised.append(min(k["lw"][1], k["rw"][1]) < min(k["ls"][1], k["rs"][1]))
        if speed:
            result.append((max(speed), np.percentile(speed, 90), max(reach), float(np.mean(raised))))
    return result


def main():
    broken = [f for xml in sorted((ROOT / "파손").rglob("*.xml")) if "val" not in str(xml.relative_to(ROOT / "파손")).lower()
              for f in features(xml)]
    zip_path = next((ROOT / "구매행동").rglob("VL_02.구매행동_05.반품.zip"))
    returns = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                with z.open(name) as fh:
                    returns += features(fh)
    names = ("max_speed", "p90_speed", "max_reach", "raised_ratio")
    for label, rows in (("파손", broken), ("반품", returns)):
        arr = np.array(rows)
        print(label, len(rows), {n: np.percentile(arr[:, i], [10, 50, 90]).round(2).tolist() for i, n in enumerate(names)})
    # 단일 기준 max_speed >= t: 파손 사람 중 비율 vs 반품 사람 중 비율
    b, r = np.array(broken)[:, 0], np.array(returns)[:, 0]
    for t in np.percentile(r, [80, 90, 95, 99]):
        print(f"max_speed>={t:.2f}: 파손 {np.mean(b >= t):.2f}  반품 {np.mean(r >= t):.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
