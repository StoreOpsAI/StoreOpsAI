import collections, csv, os, sys
os.environ["S3D_CLASSES"] = "6"
sys.path.insert(0, ".")
from pipelines import finetune_s3d as F
site = {r["clip"]: r["site"] for r in csv.DictReader(open(F.SPLIT_PATH, encoding="utf-8"))}
ok = F.audited_windows()
rows = [r for r in F.build_index() if r["label"] == 0 and (r["source"] == "normal" and r["behavior"] == "매장이동" or (r["clip"], r["start"]) in ok)]
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[r["behavior"]][site[r["clip"]]] += 1
for b, c in by.items():
    print(b, sum(c.values()), dict(c))
pos = collections.Counter(site[r["clip"]] for r in F.build_index() if r["label"] == 1)
print("양성(전체 창) 장소별", dict(pos))
