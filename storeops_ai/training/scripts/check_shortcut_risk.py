#!/usr/bin/env python3
"""
학습 전 필수 점검: '배경 shortcut' 위험 진단.

전도 모델이 웹캠에서 실패했던 원인이 바로 이것이다. 모델이 동작이 아니라
촬영 환경(매장/카메라/조명)을 외워버리는 문제.

다중 클래스에서는 더 심해진다. 예를 들어 절도는 A매장에서만, 폭행은 B매장에서만
찍혔다면, 모델은 동작을 볼 필요 없이 배경만 보고 정답을 맞힐 수 있다.
이 경우 val 정확도가 100%여도 실전에서는 전혀 작동하지 않는다.

AI-Hub 파일명 구조를 이용해 촬영 환경을 추정한다:
    C_3_7_1_BU_DYA_07-31_15-15-25_CA_RGB_DF2_M1
    ^^^^^^^^^ 앞부분 코드(장소/설정 추정)    ^^ 카메라

사용법:
    python3 check_shortcut_risk.py --data ~/dataset_train
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

KOREAN = {"abandon": "유기", "broken": "파손", "fall": "전도", "fight": "폭행",
          "fire": "방화", "smoke": "연기", "theft": "절도", "normal": "정상"}


def scene_key(source_video: str):
    """파일명에서 촬영 설정(장소/세팅)으로 추정되는 부분을 뽑는다.
    C_3_7_1_BU_DYA_07-31_... -> 'C_3_7_1_BU_DYA' (앞 5토큰)"""
    stem = Path(source_video).stem
    parts = stem.split("_")
    return "_".join(parts[:5]) if len(parts) >= 5 else stem


def date_key(source_video: str):
    """촬영 날짜(MM-DD)를 뽑는다. 같은 날 = 같은 세팅일 가능성."""
    m = re.search(r"_(\d{2}-\d{2})_", Path(source_video).stem)
    return m.group(1) if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="extract_multiclass.py 출력 폴더")
    args = ap.parse_args()

    data_dir = Path(args.data).expanduser()
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest.csv 없음: {manifest}")

    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    print(f"클립 {len(rows)}개 분석\n")

    # 클래스별 클립 수
    by_label = defaultdict(int)
    for r in rows:
        by_label[r["label"]] += 1
    print("=== 클래스별 클립 수 ===")
    for k in sorted(by_label, key=lambda x: -by_label[x]):
        print(f"  {KOREAN.get(k, k):4} : {by_label[k]}")

    # 핵심 진단: 클래스 <-> 촬영환경 교차표
    for keyname, keyfn in [("촬영설정(파일명 앞부분)", scene_key), ("촬영날짜", date_key)]:
        print(f"\n=== {keyname} x 클래스 교차 ===")
        cross = defaultdict(lambda: defaultdict(int))
        for r in rows:
            cross[keyfn(r["source_video"])][r["label"]] += 1

        labels = sorted(by_label)
        header = "  " + " ".join(f"{KOREAN.get(l, l)[:4]:>6}" for l in labels)
        print(f"{'설정':20}{header}")
        risky = []
        for scene in sorted(cross):
            counts = cross[scene]
            line = " ".join(f"{counts.get(l, 0):>6}" for l in labels)
            print(f"{scene[:20]:20}  {line}")
            # 이 설정에서 한 클래스만 나온다면 위험
            present = [l for l in labels if counts.get(l, 0) > 0 and l != "normal"]
            if len(present) == 1 and counts[present[0]] >= 3:
                risky.append((scene, present[0], counts[present[0]]))

        if risky:
            print(f"\n  [위험] 아래 설정에서는 특정 클래스만 촬영되었습니다.")
            print(f"         모델이 동작 대신 배경으로 판단할 수 있습니다.")
            for scene, lab, n in risky:
                print(f"         - {scene}: {KOREAN.get(lab, lab)}만 {n}개")
        else:
            print(f"\n  [양호] 설정별로 여러 클래스가 섞여 있습니다.")

    # 클래스별로 몇 개의 서로 다른 환경에서 촬영되었나
    print("\n=== 클래스별 촬영설정 다양성 ===")
    scenes_per_label = defaultdict(set)
    for r in rows:
        scenes_per_label[r["label"]].add(scene_key(r["source_video"]))
    for lab in sorted(scenes_per_label):
        n = len(scenes_per_label[lab])
        flag = "  <-- 환경이 1개뿐! 일반화 어려움" if n == 1 else ""
        print(f"  {KOREAN.get(lab, lab):4} : {n}개 환경{flag}")

    print("\n" + "=" * 60)
    print("해석 가이드")
    print("=" * 60)
    print("각 클래스가 여러 촬영 환경에 걸쳐 있어야 모델이 배경 대신")
    print("동작을 학습합니다. 특정 클래스가 한 환경에만 있다면:")
    print("  1) 해당 클래스 영상을 다른 환경에서 추가 확보하거나")
    print("  2) val을 '학습에 없던 환경'으로만 구성해서 실제 일반화를 측정하세요.")
    print("  3) 배경 의존을 줄이려면 포즈(스켈레톤) 기반 모델을 고려하세요.")


if __name__ == "__main__":
    main()
