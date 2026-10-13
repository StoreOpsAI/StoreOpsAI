import json
import re
from pathlib import Path
from threading import Lock


class EventManager:
    """FR-EVT-06: E + 세 자리 번호. E999 다음에는 E1000이 된다."""
    def __init__(self, event_dir):
        self.event_dir = Path(event_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def next_event_id(self):
        with self.lock:
            max_num = 0
            for p in self.event_dir.glob("E*.json"):
                m = re.fullmatch(r"E(\d{3})", p.stem)
                if m:
                    max_num = max(max_num, int(m.group(1)))
            return f"E{max_num + 1:03d}"

    def save(self, event):
        path = self.event_dir / f"{event.event_id}.json"
        path.write_text(json.dumps(event.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
