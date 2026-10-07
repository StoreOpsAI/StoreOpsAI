"""기획서 예시 값(모의 값, D17)으로 DB를 채웁니다. 모두 교육용 값이며 실제 측정값이 아닙니다."""
from __future__ import annotations

import json
import sqlite3

from .db import transaction


def seed_demo(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM order_drafts")
        conn.execute("DELETE FROM order_approvals")
        conn.execute("DELETE FROM data_coverage")

        # E014: 카메라 연결 끊김 (기획 18p) — 영상·점수 없음 (FR-EVT-12)
        conn.execute(
            """INSERT INTO events (event_id, store_id, camera_id, source, event_type, occurred_at,
               gap_sec, threshold_sec, vlm_status, status)
               VALUES ('E014','S01','CAM-01','time_rule','camera_disconnect','2026-03-14T21:04:00+09:00',
               90, 60, 'not_applicable', 'confirmed')"""
        )
        # E015: 쓰러짐 (기획 17p)
        scores = {"normal": 0.15, "fall": 0.78, "fight": 0.04, "vandalism": 0.02, "littering": 0.01}
        conn.execute(
            """INSERT INTO events (event_id, store_id, camera_id, source, event_type, occurred_at,
               clip_uri, clip_length_sec, scores, threshold, vlm_status, status)
               VALUES ('E015','S01','CAM-01','behavior_model','fall','2026-03-14T21:12:00+09:00',
               'clips/E015.mp4', 10, ?, 0.60, 'done', 'unconfirmed')""",
            (json.dumps(scores),),
        )
        # 발주 초안 (기획 23~24p, 합성 자료)
        conn.execute(
            """INSERT INTO order_drafts VALUES ('D001','S01','P001','2026-03-15',16,12,28,10,15,0,23,6,24,4,13,
               'synthetic','draft_saved',0)"""
        )
        conn.execute("INSERT INTO order_approvals VALUES ('D001',18,3,'owner-01','2026-03-14T22:05:00+09:00')")

        # 받은 기간: 3월 1일 0시 ~ 3월 16일 0시 (이 밖의 기간을 물으면 not_received)
        for rt in ("event", "order_draft"):
            conn.execute(
                "INSERT INTO data_coverage VALUES ('S01', ?, '2026-03-01T00:00:00+09:00', '2026-03-16T00:00:00+09:00')",
                (rt,),
            )
