"""Claude 시각 검수(사람 검수 아님) 결과 기록. 제외한 장면 번호와 사유, 나머지는 통과(서 있기·걷기·빈 화면).
사유: D 음료·물건을 만지거나 들어 올림, C 숙임·카운터 조작, R 사건 흔적(바닥 물건·쓰러진 의자), S 선반 앞에 서서 손 동작 판별 불가, A 바닥·의자에 앉음, E 사람이 안 보이는 빈 화면(탐지기는 사람 있다고 함)."""
import csv
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
OUT = Path("output")
EXCLUDE = {
    "SMC": {"D": [47, 48, 49, 50, 51, 52, 54, 55, 81, 82, 83, 140, 214],
            "C": [56, 59, 60, 127, 129, 132, 134, 136, 203, 204, 205, 207, 208, 210, 211, 213],
            "R": [249],
            "E": [243, 244, 245, 248, 250, 252, 256, 258, 259, 260, 281]},
    "SYB": {"R": [2, 5, 6, 8, 9, 10, 11, 42, 44, 130, 131, 132, 134]},
    "SYA": {"R": [6, 8, 10, 14, 15, 17, 18, 100, 162]},
    "SMA": {"C": [307, 310, 313, 316, 319, 429, 602, 604],
            "A": [542, 543, 546, 547, 550, 551]},
    "SMB": {"R": [20, 55, 57, 59, 99, 140],
            "A": [333, 336, 339, 523, 524, 527, 528, 531, 532, 535, 536],
            "C": [401, 403, 405, 407, 411, 413, 416, 417, 441, 443, 445, 448, 459]},
}
# 눈으로 본 시트 번호(한 장 = 20장면). 지정이 없으면 전부.
SHEETS = {"SYA": [1, 5, 9, 13],
          "SYB": [1, 3, 5, 7, 9, 11, 13, 15],
          "SMC": [1, 3, 5, 7, 9, 11, 13, 15],
          "SMA": [13, 16, 19, 22, 25, 28, 31],
          "SMB": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]}
site = sys.argv[1]
why = {i: k for k, v in EXCLUDE[site].items() for i in v}
d = SCRATCH / f"sheets_{site}_p"
rows = list(csv.DictReader(open(d / "index.csv", encoding="utf-8")))
if site in SHEETS:
    rows = [r for r in rows if (int(r["id"]) - 1) // 20 + 1 in SHEETS[site]]
with open(OUT / f"normal_audit_claude_{site}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "site", "behavior", "clip", "start", "verdict", "reason", "reviewer"])
    for r in rows:
        k = why.get(int(r["id"]))
        w.writerow([r["id"], r["site"], r["behavior"], r["clip"], r["start"], "exclude" if k else "pass", k or "", "claude_visual(not human)"])
ok = len(rows) - len(why)
print(f"{site}: 장면 {len(rows)}개, 통과 {ok}개({ok / len(rows):.0%}), 제외 {len(why)}개 {dict((k, len(v)) for k, v in EXCLUDE[site].items())}")
