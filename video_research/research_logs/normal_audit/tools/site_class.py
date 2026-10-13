import collections, csv, os, random, sys
os.environ["S3D_CLASSES"] = "6"; os.environ["AUDIT_NORMALS"] = "1"
sys.path.insert(0, ".")
from pipelines import finetune_s3d as F
site = {r["clip"]: r["site"] for r in csv.DictReader(open(F.SPLIT_PATH, encoding="utf-8"))}
rows = F.build_index(); sel = F.select_clips(rows)
it = F.epoch_items(rows, -1, sel, random.Random(0))
names = {0: "정상", 1: "전도", 2: "유기", 3: "절도", 4: "파손", 5: "방화"}
t = collections.defaultdict(collections.Counter)
for i in it:
    t[site[i["clip"]]][names[i["cls"]]] += 1
cols = list(names.values())
print("한 에폭(최종 학습) 장소 x 클래스 창 수")
print("      " + " ".join(f"{c:>5s}" for c in cols) + "   합")
for s in sorted(t):
    print(f"{s:5s} " + " ".join(f"{t[s][c]:5d}" for c in cols) + f" {sum(t[s].values()):5d}")
print("합계  " + " ".join(f"{sum(t[s][c] for s in t):5d}" for c in cols))
