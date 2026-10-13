import collections, csv, os, random, sys
sys.path.insert(0, ".")
from pipelines import finetune_s3d as F
rows = F.build_index()
print("OUT", F.OUT.name, "| fold 값", sorted({r["fold"] for r in rows}))
if F.SITE_FOLDS:
    site = {r["clip"]: r["site"] for r in csv.DictReader(open(F.SPLIT_PATH, encoding="utf-8"))}
    sel = F.select_clips(rows)
    for k in (0, 4):
        it = F.epoch_items(rows, k, sel, random.Random(0))
        held = {site[i["clip"]] for i in it}
        print(f"fold {k} 학습에 들어간 장소:", sorted(held), "| 양성", sum(i["cls"] != 0 for i in it), "정상", sum(i["cls"] == 0 for i in it))
