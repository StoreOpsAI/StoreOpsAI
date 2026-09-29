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
        # 바인드 마운트 환경에서 glob()이 새로 생긴 파일을 못 보는 경우가 있어,
        # 폴더는 시작 시 한 번만 스캔하고 이후에는 메모리 카운터로 번호를 매긴다.
        self._next_number = self._scan_max_number() + 1

    def _scan_max_number(self):
        max_num = 0
        for p in self.event_dir.glob("E*.json"):
            m = re.fullmatch(r"E(\d{3})", p.stem)
            if m:
                max_num = max(max_num, int(m.group(1)))
        return max_num

    def next_event_id(self):
        with self.lock:
            event_id = f"E{self._next_number:03d}"
            self._next_number += 1
            return event_id

    def save(self, event):
        path = self.event_dir / f"{event.event_id}.json"
        path.write_text(json.dumps(event.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, event_id):
        path = self.event_dir / f"{event_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, event_id, **changes):
        """저장된 사건의 비동기 후속 결과(VLM/영상)를 안전하게 반영한다."""
        with self.lock:
            data = self.load(event_id)
            data.update(changes)
            path = self.event_dir / f"{event_id}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data
