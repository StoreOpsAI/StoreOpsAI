import collections, os, random, sys
os.environ["S3D_CLASSES"] = "6"; os.environ["AUDIT_NORMALS"] = "1"
sys.path.insert(0, ".")
from pipelines import finetune_s3d as F
rows = F.build_index(); sel = F.select_clips(rows)
it = F.epoch_items(rows, -1, sel, random.Random(0))
neg = [i for i in it if i["cls"] == 0]; pos = [i for i in it if i["cls"] != 0]
ok = F.audited_windows()
print("양성", len(pos), "정상", len(neg), "고유 정상", len({(i["clip"], i["start"]) for i in neg}))
print("출처별 정상", dict(collections.Counter(i["behavior"] for i in neg)))
own = [i for i in neg if i["behavior"] != "매장이동"]
print("사건 영상 정상 중 검수 통과 아닌 것:", sum((i["clip"], i["start"]) not in ok for i in own))
print("OUT", F.OUT.name)
site = {}
import csv
for r in csv.DictReader(open(F.SPLIT_PATH, encoding="utf-8")): site[r["clip"]] = r["site"]
print("장소별 양성/정상/고유정상:")
for s in sorted(set(site.values())):
    p = sum(site[i["clip"]] == s for i in pos); n = [i for i in neg if site[i["clip"]] == s]
    print(f"  {s} {p} / {len(n)} / {len({(i['clip'], i['start']) for i in n})}")
