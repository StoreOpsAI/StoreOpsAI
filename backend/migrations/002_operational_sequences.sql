-- 기존 PostgreSQL 볼륨에 운영용 매장·사건 번호 시퀀스를 추가합니다.
CREATE SEQUENCE IF NOT EXISTS store_number_seq;
CREATE SEQUENCE IF NOT EXISTS event_number_seq;

SELECT setval(
    'store_number_seq',
    GREATEST(
        COALESCE((SELECT MAX(CAST(SUBSTRING(store_id FROM 2) AS bigint)) FROM stores WHERE store_id ~ '^S[0-9]+$'), 0),
        1
    ),
    true
);

SELECT setval(
    'event_number_seq',
    GREATEST(
        COALESCE((SELECT MAX(CAST(SUBSTRING(event_id FROM 2) AS bigint)) FROM events WHERE event_id ~ '^E[0-9]+$'), 0),
        1
    ),
    true
);
