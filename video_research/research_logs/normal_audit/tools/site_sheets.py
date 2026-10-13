"""장소 하나의 정상 후보 창(사건 밖, 직전·직후 16초 제외)을 10초 간격으로 뽑아 검수 그림을 만든다. 창마다 시작·가운데·끝 3프레임, 한 장에 20개(2열x10행)."""
import csv
import os
import pickle
import sys
from pathlib import Path

os.environ["S3D_CLASSES"] = "6"
os.environ["PRE_EVENT_MARGIN"] = "48"
os.environ["POST_EVENT_MARGIN"] = "48"
sys.path.insert(0, ".")
import cv2
import numpy as np

from pipelines import finetune_s3d as F

SITE = os.environ["SITE"]
STEP = int(os.getenv("STEP", "30"))   # 표본(3fps) 간격: 30 = 10초
OUT = Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
site = {r["clip"]: r["site"] for r in csv.DictReader(open(F.SPLIT_PATH, encoding="utf-8"))}
rows = [r for r in F.build_index() if r["label"] == 0 and r["behavior"] != "매장이동" and site.get(r["clip"]) == SITE
        and not r.get("pre_event") and not r.get("post_event")]
rows.sort(key=lambda r: (r["clip"], r["start"]))
picked, last = [], {}
for r in rows:
    if r["start"] - last.get(r["clip"], -10**9) >= STEP:
        picked.append(r); last[r["clip"]] = r["start"]

cache = {}
def frames(clip, start):
    if clip not in cache:
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        cache[clip] = pickle.loads((F.FRAMES / f"{clip}.pkl").read_bytes())["jpeg"]
    data = cache[clip]
    return [cv2.resize(cv2.imdecode(np.frombuffer(data[min(start + k, len(data) - 1)], np.uint8), cv2.IMREAD_COLOR), (320, 180)) for k in (0, 6, 11)]

from ultralytics import YOLO
det = YOLO(str(Path(F.APP_ROOT) / "yolo11s-pose.pt")) if hasattr(F, "APP_ROOT") else YOLO("yolo11s-pose.pt")
def has_person(clip, start):   # 보여 주는 3프레임 중 하나라도 사람 상자(기본 신뢰도 0.25)가 있으면 사람 있음
    data = pickle.loads((F.FRAMES / f"{clip}.pkl").read_bytes())["jpeg"] if clip not in cache else cache[clip]
    ims = [cv2.imdecode(np.frombuffer(data[min(start + k, len(data) - 1)], np.uint8), cv2.IMREAD_COLOR) for k in (0, 6, 11)]
    return any(len(r.boxes) for r in det(ims, verbose=False))
if os.getenv("PERSON_ONLY") == "1":
    kept = [r for r in picked if has_person(r["clip"], r["start"])]
    print(f"{SITE}: 10초 장면 {len(picked)}개 중 사람 없음 {len(picked) - len(kept)}개 제외")
    json_dump = {"empty": [(r["clip"], r["start"]) for r in picked if r not in kept]}
    import json; (OUT / "empty.json").write_text(json.dumps(json_dump), encoding="utf-8")
    picked = kept

tiles, index = [], []
for i, r in enumerate(picked, 1):
    strip = np.hstack(frames(r["clip"], r["start"]))
    cv2.putText(strip, f"#{i}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    tiles.append(strip)
    index.append({"id": i, "site": SITE, "behavior": r["behavior"], "clip": r["clip"], "start": r["start"]})
blank = np.zeros_like(tiles[0])
n = (len(tiles) + 19) // 20
for s in range(n):
    chunk = tiles[s * 20:(s + 1) * 20]
    chunk += [blank] * (20 - len(chunk))
    sheet = np.vstack([np.hstack(chunk[j * 2:(j + 1) * 2]) for j in range(10)])
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(str(OUT / f"sheet_{s + 1:03d}.jpg"))
with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "site", "behavior", "clip", "start"]); w.writeheader(); w.writerows(index)
print(f"{SITE}: 후보 창 {len(rows)}개 -> 10초 간격 {len(picked)}장면, 그림 {n}장 -> {OUT}")
