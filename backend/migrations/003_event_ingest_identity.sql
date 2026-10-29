-- 기존 사건은 보존하고 새 탐지 사건부터 UUID로 중복 등록을 방지합니다.
ALTER TABLE events ADD COLUMN IF NOT EXISTS event_uid uuid;
CREATE UNIQUE INDEX IF NOT EXISTS uq_events_ingest
    ON events (store_id, camera_id, event_uid);
