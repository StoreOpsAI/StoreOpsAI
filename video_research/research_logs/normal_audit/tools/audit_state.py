"""검수 상태(본 시트, 제외 장면과 사유)를 audit_state.json에 쌓고, 장소별 CSV를 만든다. 사용: python audit_state.py SITE 시트번호,시트번호 "R:1,2;C:5" / python audit_state.py SITE build"""
import csv, json, sys
from pathlib import Path
S = Path(__file__).parent
ST = S / "audit_state.json"
OUT = Path("output")
state = json.loads(ST.read_text(encoding="utf-8")) if ST.exists() else {}
site, sheets = sys.argv[1], sys.argv[2]
st = state.setdefault(site, {"sheets": [], "exclude": {}})
if sheets != "build":
    for n in sheets.split(","):
        if int(n) not in st["sheets"]:
            st["sheets"].append(int(n))
    for part in (sys.argv[3].split(";") if len(sys.argv) > 3 and sys.argv[3] else []):
        k, ids = part.split(":")
        for i in ids.split(","):
            if i:
                st["exclude"][i.strip()] = k
    ST.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
rows = [r for r in csv.DictReader(open(S / f"sheets_{site}_p" / "index.csv", encoding="utf-8")) if (int(r["id"]) - 1) // 20 + 1 in st["sheets"]]
with open(OUT / f"normal_audit_claude_{site}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "site", "behavior", "clip", "start", "verdict", "reason", "reviewer"])
    for r in rows:
        k = st["exclude"].get(r["id"])
        w.writerow([r["id"], r["site"], r["behavior"], r["clip"], r["start"], "exclude" if k else "pass", k or "", "claude_visual(not human)"])
p = sum(1 for r in rows if r["id"] not in st["exclude"])
n_all = len(list(csv.DictReader(open(S / f"sheets_{site}_p" / "index.csv", encoding="utf-8"))))
print(f"{site}: 본 시트 {len(st['sheets'])}장, 본 장면 {len(rows)}/{n_all}, 통과 {p}")
