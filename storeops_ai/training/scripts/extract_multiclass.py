#!/usr/bin/env python3
"""
다중 이벤트 클래스(쓰러짐/싸움/파손/쓰레기 투기 등) 프레임 추출기.

이벤트 타입마다 동작의 시간 특성이 달라서 자르는 방식을 다르게 한다.
단, 출력 프레임 수는 모든 클래스가 동일해야 한다(배치 학습 요건).

  motion_split : 앞부분에 dense, 뒷부분에 sparse 배분.
                 '동작 후 정지'하는 이벤트(전도)에 적합.
  uniform      : 이벤트 구간 전체를 균일 리샘플. 짧고 단발성인 이벤트에 적합.
  sliding      : 짧은 윈도우를 겹쳐가며 여러 클립 생성.
                 길고 반복적인 이벤트(폭행/절도)에 적합. 데이터 증강 효과도 있음.

사용법:
    python3 extract_multiclass.py --videos ~/aihub_all --labels ~/aihub_all \
        --out ~/dataset_train --data-source aihub_public --frames 24 --extract_normals

    # 타입별 설정을 바꾸려면
    python3 extract_multiclass.py ... --config my_config.json
"""

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

EVENT_PREFIXES = ["abandon", "broken", "fall", "fight", "fire", "smoke", "theft"]

KOREAN = {"abandon": "쓰레기 투기", "broken": "파손", "fall": "쓰러짐", "fight": "싸움",
          "fire": "방화", "smoke": "연기", "theft": "절도", "normal": "정상"}

CORE_KEYPOINTS = {"Pelvis", "Spine naval", "Spine chest", "Neck base", "Center head"}

# 타입별 기본 전략. analyze_event_durations.py 결과를 보고 조정할 것.
# max_clips: 영상 1개당 만들 클립 수 상한(클래스별).
# 생략하면 --max_clips_per_video 값을 쓴다. sliding 모드는 클립이 여러 개
# 나오므로, 다른 클래스(영상당 1개)와 균형을 맞추려면 여기서 낮춰야 한다.
DEFAULT_CONFIG = {
    "fall":    {"mode": "motion_split", "padding": 15, "dense_ratio": 0.667, "max_clips": 1},
    "fight":   {"mode": "sliding", "padding": 10, "window_sec": 1.5, "step_sec": 0.75,
                "max_clips": 3},
    "theft":   {"mode": "sliding", "padding": 15, "window_sec": 3.0, "step_sec": 1.5,
                "max_clips": 1},
    "broken":  {"mode": "uniform", "padding": 10, "max_clips": 1},
    "fire":    {"mode": "uniform", "padding": 30, "max_clips": 1},
    "abandon": {"mode": "uniform", "padding": 30, "max_clips": 1},
    "smoke":   {"mode": "uniform", "padding": 30, "max_clips": 1},
}


def pick_spread(items, k):
    """items에서 k개를 앞쪽에 몰리지 않게 고루 고른다.
    k=1이면 (처음이 아니라) 가운데를 고른다 - 이벤트 중반이 가장 대표적이므로."""
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 1:
        return [items[n // 2]]
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    return [items[j] for j in sorted(set(idx.tolist()))]


# ------------------------------------------------------------------ 라벨 파싱

# ---------------------------------------------------- NIA2019 형식 (다른 출처)
# AI-Hub CVAT과 달리 root가 <annotation>(단수)이고, 이벤트 구간 대신
# <object><action><frame><start>/<end> 로 '동작 구간'이 직접 라벨링되어 있다.
# 이벤트 전체(수십 초)보다 액션 구간(1~8초)이 훨씬 정확한 학습 구간이다.

NIA_EVENT_MAP = {
    "assault": "fight", "fight": "fight", "datefight": "fight",
    "theft": "theft", "burglary": "theft", "robbery": "theft",
    "vandalism": "broken", "dump": "abandon", "swoon": "fall",
}

# 해당 클래스에서 '그 동작이 아닌' 것으로 제외할 actionname
NIA_ACTION_EXCLUDE = {
    "fight": {"falldown"},   # 맞아서 쓰러지는 건 폭행 동작이 아니라 결과
}


def _timecode_to_frames(tc: str, fps: float):
    """'00:03:47.2' -> 프레임 번호"""
    hh, mm, ss = tc.split(":")
    return int(round((int(hh) * 3600 + int(mm) * 60 + float(ss)) * fps))


def _merge_spans(spans, gap=0):
    """겹치거나 붙어있는 (start,end) 구간을 합친다."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _parse_nia_character(header):
    """<character>F40,F40</character> -> (gender, age_group)"""
    ch = (header.findtext("character") or "").strip()
    if not ch:
        return None, None
    first = ch.split(",")[0].strip().upper()
    gender = {"M": "male", "F": "female"}.get(first[:1])
    age = None
    digits = "".join(c for c in first[1:] if c.isdigit())
    if digits:
        d = int(digits)
        age = ("child" if d < 20 else "young people" if d < 40
               else "middle age" if d < 60 else "advanced age")
    return gender, age


def parse_nia2019(xml_path: Path, use_falldown=False, min_frames=15, seen_actions=None):
    """NIA2019 XML에서 학습 구간 목록을 뽑는다. CVAT과 달리 여러 구간이 나온다."""
    root = ET.parse(xml_path).getroot()
    header = root.find("header")
    if header is None:
        return []

    fps = float(header.findtext("fps") or 30.0)
    total = int(header.findtext("frames") or 0)
    gender, age = _parse_nia_character(header)

    ev = root.find("event")
    raw_name = (ev.findtext("eventname") if ev is not None else "") or ""
    label = NIA_EVENT_MAP.get(raw_name.strip().lower())
    if label is None:
        return []

    exclude = NIA_ACTION_EXCLUDE.get(label, set())

    by_label = defaultdict(list)   # 최종 라벨 -> [(start,end), ...]
    for obj in root.findall("object"):
        for act in obj.findall("action"):
            aname = (act.findtext("actionname") or "").strip().lower()
            if seen_actions is not None:
                seen_actions[f"{raw_name}:{aname}"] += 1

            spans = []
            for fr in act.findall("frame"):
                try:
                    s = int(fr.findtext("start")); e = int(fr.findtext("end"))
                except (TypeError, ValueError):
                    continue
                if e > s:
                    spans.append((s, e))
            if not spans:
                continue

            if aname in exclude:
                # falldown은 원하면 전도 데이터로 따로 쓸 수 있다
                if use_falldown and aname == "falldown":
                    by_label["fall"].extend(spans)
                continue
            by_label[label].extend(spans)

    out = []
    for lab, spans in by_label.items():
        for s, e in _merge_spans(spans, gap=int(fps * 0.5)):
            if e - s + 1 < min_frames:
                continue   # 너무 짧은 구간은 버린다
            if total and e >= total:
                e = total - 1
            if e <= s:
                continue
            out.append({"event_type": lab, "start_frame": s, "end_frame": e,
                        "gender": gender, "age_group": age, "source_format": "nia2019"})
    return out


def detect_format(xml_path: Path):
    """root 태그로 형식을 구분한다. CVAT은 <annotations>, NIA2019는 <annotation>."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return "invalid"
    return {"annotations": "cvat", "annotation": "nia2019"}.get(root.tag, "unknown")


def find_event_spans(xml_path: Path, use_falldown=False, seen_actions=None):
    """형식을 자동 판별해서 학습 구간 목록을 반환한다 (항상 리스트)."""
    fmt = detect_format(xml_path)
    if fmt == "nia2019":
        return parse_nia2019(xml_path, use_falldown=use_falldown, seen_actions=seen_actions)
    span = find_event_span(xml_path)
    if span is None:
        return []
    span["source_format"] = "cvat"
    return [span]


def find_event_span(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks = defaultdict(list)
    for t in root.findall("track"):
        tracks[t.get("label")].append(t)

    for prefix in EVENT_PREFIXES:
        starts, ends = tracks.get(f"{prefix}_start"), tracks.get(f"{prefix}_end")
        if not starts or not ends:
            continue
        sf = [int(b.get("frame")) for t in starts for b in t.findall("box")
              if b.get("outside") == "0"]
        ef = [int(b.get("frame")) for t in ends for b in t.findall("box")
              if b.get("outside") == "0"]
        if not sf or not ef:
            continue

        gender = age = None
        for t in starts:
            for b in t.findall("box"):
                for a in b.findall("attribute"):
                    if a.get("name") == "gender" and a.text:
                        gender = a.text.strip()
                    elif a.get("name") == "age group" and a.text:
                        age = a.text.strip()

        return {"event_type": prefix, "start_frame": min(sf), "end_frame": max(ef),
                "gender": gender, "age_group": age}
    return None


def get_motion_profile(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pts_by_frame = defaultdict(list)
    for track in root.findall("track"):
        if track.get("label") not in CORE_KEYPOINTS:
            continue
        for pt in track.findall("points"):
            if pt.get("outside") == "1":
                continue
            x, y = map(float, pt.get("points").split(","))
            pts_by_frame[int(pt.get("frame"))].append((x, y))

    frames = sorted(pts_by_frame)
    cent = {f: (sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v))
            for f, v in pts_by_frame.items()}
    motion = {}
    for a, b in zip(frames, frames[1:]):
        (x0, y0), (x1, y1) = cent[a], cent[b]
        motion[b] = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return motion


def find_settle_frame(motion, w_start, w_end, ev_start, ev_end, buffer_frames=2):
    vals = [(f, v) for f, v in motion.items() if w_start <= f <= w_end]
    if len(vals) < 3:
        return ev_end
    ordered = sorted(v for _, v in vals)
    baseline = ordered[len(ordered) // 2]
    thr = max(baseline * 2.5, baseline + 1.0)
    big = [f for f, v in vals if v > thr]
    if not big:
        return ev_end
    return max(ev_start, min(max(big) + buffer_frames, w_end))


# ------------------------------------------------------------------ 프레임 추출

def _save_frames(video_path: Path, target_frames, out_dir: Path):
    """지정된 프레임 인덱스 목록(정렬됨, 중복 허용)을 순서대로 저장."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start = int(target_frames[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    saved = 0
    ptr = 0
    cur = start
    last = None
    n = len(target_frames)
    while cur <= int(target_frames[-1]) and ptr < n:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        while ptr < n and int(target_frames[ptr]) == cur:
            saved += 1
            cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            ptr += 1
        cur += 1
    while ptr < n and last is not None:
        saved += 1
        cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), last,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        ptr += 1
    cap.release()
    return saved, fps


def plan_uniform(w_start, w_end, n_frames):
    return np.round(np.linspace(w_start, w_end, n_frames)).astype(int)


def plan_motion_split(w_start, w_end, settle, n_frames, dense_ratio):
    dense_n = max(1, int(round(n_frames * dense_ratio)))
    sparse_n = n_frames - dense_n
    settle = max(w_start, min(settle, w_end))

    dense = np.round(np.linspace(w_start, settle, dense_n)).astype(int)
    if sparse_n > 0:
        s0 = min(settle + 1, w_end)
        sparse = np.round(np.linspace(s0, w_end, sparse_n)).astype(int)
        return np.concatenate([dense, sparse])
    return dense


def plan_sliding(w_start, w_end, n_frames, fps, window_sec, step_sec):
    """겹치는 윈도우 여러 개를 만들어 각각의 프레임 계획을 리스트로 반환."""
    win = max(n_frames, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    span = w_end - w_start + 1

    if span <= win:
        return [plan_uniform(w_start, w_end, n_frames)]

    plans = []
    s = w_start
    while s + win - 1 <= w_end:
        plans.append(plan_uniform(s, s + win - 1, n_frames))
        s += step
    # 마지막 구간이 남으면 끝에 맞춘 윈도우 하나 추가
    if plans is not None and (w_end - (s - step + win - 1)) > step // 2:
        plans.append(plan_uniform(w_end - win + 1, w_end, n_frames))
    return plans


def get_fps(video_path: Path, default=30.0):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (fps if fps and fps > 1 else default), total


# ------------------------------------------------------------------ 메인

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", default=None, help="기본: --videos와 동일")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-source", required=True,
                     choices=["aihub_public", "team_consented"],
                     help="허용된 원본 영상 출처")
    ap.add_argument("--consent-confirmed", action="store_true",
                     help="팀원 연기 영상의 촬영·학습 동의를 확인했음을 표시")
    ap.add_argument("--frames", type=int, default=24,
                     help="클립당 출력 프레임 수 (모든 클래스 공통)")
    ap.add_argument("--config", default=None, help="타입별 설정 JSON (생략 시 기본값)")
    ap.add_argument("--extract_normals", action="store_true")
    ap.add_argument("--normals_per_video", type=int, default=1)
    ap.add_argument("--normal_buffer", type=int, default=20)
    ap.add_argument("--normal_window_sec", type=float, default=4.0,
                     help="정상 클립이 커버할 시간 길이(초). 이벤트 클립과 비슷해야 한다")
    ap.add_argument("--normal_video_ratio", type=float, default=1.0,
                     help="정상 클립을 뽑을 영상 비율(0~1). 정상 클래스가 너무 많아지는 걸 방지")
    ap.add_argument("--max_clips_per_video", type=int, default=6,
                     help="sliding 모드에서 영상 1개당 만들 최대 클립 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_classes", default="",
                     help="학습에서 뺄 클래스(쉼표 구분). 예: fight")
    args = ap.parse_args()

    if args.data_source == "team_consented" and not args.consent_confirmed:
        ap.error("팀원 연기 영상은 동의를 확인한 경우에만 --consent-confirmed를 지정하세요.")

    exclude_classes = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    if exclude_classes:
        print(f"제외 클래스: {', '.join(sorted(exclude_classes))}")

    videos_dir = Path(args.videos).expanduser()
    labels_dir = Path(args.labels).expanduser() if args.labels else videos_dir
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    if args.config:
        user_cfg = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
        for k, v in user_cfg.items():
            config.setdefault(k, {}).update(v)

    rng = random.Random(args.seed)
    xml_files = sorted(labels_dir.glob("*.xml"))
    print(f"라벨 {len(xml_files)}개 발견 | 클립당 {args.frames}프레임")

    rows = []
    stats = defaultdict(int)
    fmt_stats = defaultdict(int)
    seen_actions = defaultdict(int)
    n_no_video = n_no_event = n_excluded = 0

    for i, xml_path in enumerate(xml_files, 1):
        stem = xml_path.stem
        video_path = videos_dir / f"{stem}.mp4"
        if not video_path.exists():
            n_no_video += 1
            continue

        if detect_format(xml_path) != "cvat":
            print(f"[제외] 허용된 AI Hub/팀원 CVAT 라벨 형식이 아닙니다: {stem}")
            n_excluded += 1
            continue

        spans = find_event_spans(xml_path, seen_actions=seen_actions)
        if not spans:
            print(f"[경고] 이벤트 없음: {stem} (형식={detect_format(xml_path)})")
            n_no_event += 1
            continue

        # 제외 클래스 걸러내기 (예: 출처가 달라 shortcut 위험이 큰 폭행)
        spans = [s for s in spans if s["event_type"] not in exclude_classes]
        if not spans:
            n_excluded += 1
            continue

        fmt_stats[spans[0].get("source_format", "?")] += 1
        fps, total = get_fps(video_path)
        made_here = 0   # 이 영상에서 만든 클립 수 (여러 구간 합산으로 제한)

        # 이 영상의 클립 수 상한 (클래스별 설정 > 전역 옵션)
        cap = config.get(spans[0]["event_type"], {}).get(
            "max_clips", args.max_clips_per_video)

        # 구간이 상한보다 많으면 앞쪽만 쓰지 말고 전체에 고루 퍼지게 고른다.
        # (폭행 영상에서 초반 pushing만 뽑히고 후반 punching을 놓치는 걸 방지)
        spans = pick_spread(spans, cap)

        for si, span in enumerate(spans):
            if made_here >= cap:
                break

            etype = span["event_type"]
            cfg = config.get(etype, {"mode": "uniform", "padding": 15})

            pad = cfg.get("padding", 15)
            w_start = max(0, span["start_frame"] - pad)
            w_end = (min(total - 1, span["end_frame"] + pad) if total > 0
                     else span["end_frame"] + pad)
            if w_end <= w_start:
                print(f"[경고] 구간 이상: {stem}")
                continue

            mode = cfg.get("mode", "uniform")
            if mode == "motion_split" and span.get("source_format") == "cvat":
                motion = get_motion_profile(xml_path)
                settle = find_settle_frame(motion, w_start, w_end,
                                            span["start_frame"], span["end_frame"])
                plans = [plan_motion_split(w_start, w_end, settle, args.frames,
                                            cfg.get("dense_ratio", 0.667))]
                settle_info = settle
            elif mode == "sliding":
                plans = plan_sliding(w_start, w_end, args.frames, fps,
                                      cfg.get("window_sec", 2.0), cfg.get("step_sec", 1.0))
                settle_info = None
            else:
                # NIA는 스켈레톤이 없어 motion_split 불가 -> uniform으로 대체
                plans = [plan_uniform(w_start, w_end, args.frames)]
                settle_info = None

            # 남은 여유만큼만, 그리고 구간 전체에 고루 퍼지게 고른다
            plans = pick_spread(plans, max(1, cap - made_here))
            made_here += len(plans)
            multi = len(spans) > 1 or len(plans) > 1

            for k, plan in enumerate(plans):
                suffix = (f"__{etype}" if not multi
                          else f"__{etype}_s{si+1}_{k+1}")
                clip_name = stem + suffix
                clip_out = out_dir / clip_name
                n_saved, _ = _save_frames(video_path, plan, clip_out)

                meta = {
                    "event_type": etype, "label": etype, "mode": mode,
                    "source_format": span.get("source_format", "cvat"),
                    "gender": span["gender"], "age_group": span["age_group"],
                    "raw_start_frame": span["start_frame"],
                    "raw_end_frame": span["end_frame"],
                    "window_start": int(plan[0]), "window_end": int(plan[-1]),
                    "settle_frame": settle_info, "num_frames": n_saved, "fps": fps,
                    "source_video": video_path.name,
                    "data_source": args.data_source,
                    "consent_confirmed": str(args.consent_confirmed).lower(),
                }
                (clip_out / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append({"clip_folder": clip_name, **meta})
                stats[etype] += 1

        # ---- 정상 클립 ----
        span = spans[0]
        if args.extract_normals and rng.random() <= args.normal_video_ratio:
            n_end = span["start_frame"] - args.normal_buffer
            if n_end >= args.frames:
                # 정상 클립도 이벤트 클립과 비슷한 '시간 길이'여야 한다.
                # 가용 구간 전체를 24프레임에 욱여넣으면 (NIA처럼 이벤트 전이
                # 몇 분씩 되는 경우) 프레임 간격이 수 초가 되어 동작이 아니게 된다.
                win = max(args.frames, int(round(args.normal_window_sec * fps)))
                seg_len = (n_end + 1) // max(1, args.normals_per_video)
                for k in range(args.normals_per_video):
                    lo = k * seg_len
                    hi = min(lo + seg_len - 1, n_end)
                    if hi - lo + 1 < 4:
                        continue
                    # 구간이 길면 그 안에서 win 길이만큼 무작위 위치로 자른다
                    if hi - lo + 1 > win:
                        s0 = rng.randint(lo, hi - win + 1)
                        s1 = s0 + win - 1
                    else:
                        s0, s1 = lo, hi
                    # 전도 클립과 시간 구조를 맞추기 위해 동일한 motion_split 형태 사용
                    pseudo = s0 + int((s1 - s0) * rng.uniform(0.25, 0.75))
                    plan = plan_motion_split(s0, s1, pseudo, args.frames, 0.667)

                    clip_name = f"{stem}__normal_{k+1}"
                    clip_out = out_dir / clip_name
                    n_saved, _ = _save_frames(video_path, plan, clip_out)

                    meta = {
                        "event_type": "normal", "label": "normal", "mode": "normal",
                        "source_format": span.get("source_format", "cvat"),
                        "gender": span["gender"], "age_group": span["age_group"],
                        "raw_start_frame": s0, "raw_end_frame": s1,
                        "window_start": int(plan[0]), "window_end": int(plan[-1]),
                        "settle_frame": pseudo, "num_frames": n_saved, "fps": fps,
                        "source_video": video_path.name,
                        "data_source": args.data_source,
                        "consent_confirmed": str(args.consent_confirmed).lower(),
                    }
                    (clip_out / "meta.json").write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    rows.append({"clip_folder": clip_name, **meta})
                    stats["normal"] += 1

        if i % 25 == 0:
            print(f"  ...{i}/{len(xml_files)}")

    if not rows:
        raise SystemExit("생성된 클립이 없습니다.")

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n영상없음 {n_no_video} / 이벤트못찾음 {n_no_event} / 제외클래스 {n_excluded}")

    if fmt_stats:
        print("\n=== 라벨 형식별 영상 수 ===")
        for k in sorted(fmt_stats):
            print(f"  {k:8} : {fmt_stats[k]}")
    print("\n=== 클래스별 클립 수 ===")
    for k in sorted(stats, key=lambda x: -stats[x]):
        print(f"  {KOREAN.get(k, k):4} ({k:7}) : {stats[k]}")
    print(f"  합계: {sum(stats.values())}")

    lo, hi = min(stats.values()), max(stats.values())
    if hi > lo * 3:
        print(f"\n[주의] 클래스 불균형이 큽니다 ({lo} ~ {hi}). "
              f"--normal_video_ratio 나 --max_clips_per_video 로 조절하세요.")

    print(f"\n결과: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
