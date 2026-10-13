"""행동 공통 OOF 파라미터 탐색 (스펙 02 §4~6, §11, §18).

evaluate(**params) -> {"recall", "false_alarm_per_hour", ...}를 넘기면 후보 조합 전체를 평가해 threshold_search.csv로 저장하고,
Event Recall 제약마다 False Alarm/Hour가 가장 낮은 조합을 selected_params.json에 기록한다. 최종값 하나만 남기지 않는다.
후보값은 임의 격자가 아니라 OOF 점수·관측 분포의 고유값/분위수에서 만든다(candidates_from_values).
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


def candidates_from_values(values: Sequence[float], max_count: int = 200) -> list[float]:
    """관측 값의 고유값. max_count를 넘으면 분위수로 줄인다(개수 상한은 계산량 때문이며 값은 데이터에서만 나온다)."""
    unique = np.unique(np.asarray(values, dtype=float))
    if len(unique) <= max_count:
        return [float(v) for v in unique]
    return [float(v) for v in np.unique(np.quantile(unique, np.linspace(0, 1, max_count)))]


def search_params(evaluate: Callable[..., Mapping[str, float]], space: Mapping[str, Sequence[float]],
                  recall_targets: Sequence[float], out_dir: Path, meta: Mapping[str, object]) -> dict:
    """space의 모든 조합을 평가해 out_dir에 threshold_search.csv, selected_params.json을 쓴다. 선택 결과를 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(space)
    rows = [{**dict(zip(names, combo)), **evaluate(**dict(zip(names, combo)))} for combo in itertools.product(*space.values())]
    with open(out_dir / "threshold_search.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) + [f"ok_recall_{t}" for t in recall_targets])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{f"ok_recall_{t}": row["recall"] >= t for t in recall_targets}})
    selected = {}
    for target in recall_targets:
        ok = [r for r in rows if r["recall"] >= target]
        # 목표 재현율을 만족하는 조합 중 오경보/시간 최소, 같으면 재현율 높은 쪽
        best = min(ok, key=lambda r: (r["false_alarm_per_hour"], -r["recall"]), default=None)
        selected[str(target)] = best and {k: best[k] for k in best}
    report = {"objective": "min_false_alarm_per_hour_under_recall_constraint", "search_space_sizes": {n: len(v) for n, v in space.items()},
              "selected_by_recall_target": selected, "test_used_for_selection": False, **meta}
    (out_dir / "selected_params.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    import tempfile

    cands = candidates_from_values([0.1, 0.1, 0.4, 0.9])
    assert cands == [0.1, 0.4, 0.9]
    fake = lambda thr: {"recall": 1.0 - thr, "false_alarm_per_hour": 10 * (1 - thr) ** 2}  # 높을수록 재현율↓ 오경보↓
    with tempfile.TemporaryDirectory() as d:
        rep = search_params(fake, {"thr": [0.1, 0.4, 0.9]}, (0.5, 0.95), Path(d), {})
        assert rep["selected_by_recall_target"]["0.5"]["thr"] == 0.4  # 재현율 0.6 >= 0.5 중 오경보 최소
        assert rep["selected_by_recall_target"]["0.95"] is None  # 만족하는 조합 없음
        assert (Path(d) / "threshold_search.csv").is_file()
    print("ok")
