-- StoreOps AI 초기 PostgreSQL 스키마입니다.
-- pgvector 임베딩 차원과 실제 파일 저장소는 결정 후 별도 마이그레이션으로 추가합니다.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SEQUENCE store_number_seq;
CREATE SEQUENCE event_number_seq;

CREATE TABLE stores (
    store_id varchar(32) PRIMARY KEY,
    name varchar(120) NOT NULL,
    timezone varchar(64) NOT NULL DEFAULT 'Asia/Seoul',
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    user_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id varchar(32) NOT NULL REFERENCES stores(store_id),
    email varchar(255) NOT NULL UNIQUE,
    password_hash text NOT NULL,
    display_name varchar(120) NOT NULL,
    role varchar(16) NOT NULL DEFAULT 'owner'
        CHECK (role = 'owner'),
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE auth_sessions (
    session_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_token_hash text NOT NULL UNIQUE,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    revoked_at timestamptz
);

CREATE TABLE cameras (
    camera_id varchar(32) PRIMARY KEY,
    store_id varchar(32) NOT NULL REFERENCES stores(store_id),
    name varchar(120) NOT NULL,
    is_connected boolean NOT NULL DEFAULT false,
    last_frame_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE events (
    event_id varchar(16) PRIMARY KEY,
    store_id varchar(32) NOT NULL REFERENCES stores(store_id),
    camera_id varchar(32) NOT NULL REFERENCES cameras(camera_id),
    source varchar(32) NOT NULL
        CHECK (source IN ('behavior_model', 'time_rule')),
    event_type varchar(32) NOT NULL,
    occurred_at timestamptz NOT NULL,
    status varchar(16) NOT NULL DEFAULT 'unconfirmed'
        CHECK (status IN ('unconfirmed', 'confirmed', 'resolved', 'false_alarm')),
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    threshold numeric(5,4) CHECK (threshold BETWEEN 0 AND 1),
    gap_sec integer CHECK (gap_sec >= 0),
    threshold_sec integer CHECK (threshold_sec >= 0),
    data_label varchar(16) NOT NULL DEFAULT 'mock'
        CHECK (data_label IN ('real', 'synthetic', 'mock')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (source = 'time_rule' AND event_type = 'camera_disconnect'
            AND scores = '{}'::jsonb AND threshold IS NULL
            AND gap_sec IS NOT NULL AND threshold_sec IS NOT NULL)
        OR
        (source = 'behavior_model' AND event_type <> 'camera_disconnect'
            AND gap_sec IS NULL AND threshold_sec IS NULL)
    )
);

CREATE TABLE event_media (
    media_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id varchar(16) NOT NULL REFERENCES events(event_id),
    media_type varchar(16) NOT NULL
        CHECK (media_type IN ('clip', 'frame_start', 'frame_middle', 'frame_end')),
    uri text NOT NULL,
    length_sec integer CHECK (length_sec >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (event_id, media_type)
);

CREATE TABLE vlm_results (
    event_id varchar(16) PRIMARY KEY REFERENCES events(event_id),
    status varchar(20) NOT NULL
        CHECK (status IN ('pending', 'completed', 'failed', 'not_applicable')),
    model_name varchar(120),
    observed text,
    uncertain text,
    owner_checks text,
    missing_fields jsonb NOT NULL DEFAULT '[]'::jsonb,
    error_message text,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE event_status_history (
    history_id bigserial PRIMARY KEY,
    event_id varchar(16) NOT NULL REFERENCES events(event_id),
    from_status varchar(16) NOT NULL,
    to_status varchar(16) NOT NULL,
    changed_by uuid NOT NULL REFERENCES users(user_id),
    changed_at timestamptz NOT NULL DEFAULT now(),
    reason text
);

CREATE TABLE products (
    product_id varchar(64) PRIMARY KEY,
    product_name varchar(160) NOT NULL,
    unit_name varchar(32) NOT NULL,
    default_pack_size integer NOT NULL CHECK (default_pack_size > 0),
    data_label varchar(16) NOT NULL DEFAULT 'mock'
        CHECK (data_label IN ('real', 'synthetic', 'mock')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE order_drafts (
    draft_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id varchar(32) NOT NULL REFERENCES stores(store_id),
    product_id varchar(64) NOT NULL REFERENCES products(product_id),
    target_date date NOT NULL,
    forecast_d1 numeric(12,2) NOT NULL CHECK (forecast_d1 >= 0),
    forecast_d2 numeric(12,2) NOT NULL CHECK (forecast_d2 >= 0),
    forecast_total numeric(12,2) NOT NULL CHECK (forecast_total >= 0),
    safety_stock numeric(12,2) NOT NULL CHECK (safety_stock >= 0),
    on_hand numeric(12,2) NOT NULL CHECK (on_hand >= 0),
    incoming numeric(12,2) NOT NULL CHECK (incoming >= 0),
    need numeric(12,2) NOT NULL CHECK (need >= 0),
    pack_size integer NOT NULL CHECK (pack_size > 0),
    recommended_qty integer NOT NULL CHECK (recommended_qty >= 0),
    recommended_boxes integer NOT NULL CHECK (recommended_boxes >= 0),
    shortage_before_arrival numeric(12,2) NOT NULL CHECK (shortage_before_arrival >= 0),
    status varchar(16) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'cancelled')),
    data_label varchar(16) NOT NULL DEFAULT 'mock'
        CHECK (data_label IN ('real', 'synthetic', 'mock')),
    sent_to_supplier boolean NOT NULL DEFAULT false
        CHECK (sent_to_supplier = false),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (store_id, product_id, target_date)
);

CREATE TABLE order_approvals (
    approval_id bigserial PRIMARY KEY,
    draft_id uuid NOT NULL UNIQUE REFERENCES order_drafts(draft_id),
    approved_qty integer NOT NULL CHECK (approved_qty >= 0),
    approved_boxes integer NOT NULL CHECK (approved_boxes >= 0),
    approved_by uuid NOT NULL REFERENCES users(user_id),
    approved_at timestamptz NOT NULL DEFAULT now(),
    note text
);

CREATE TABLE qa_sessions (
    session_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id varchar(32) NOT NULL REFERENCES stores(store_id),
    asked_by uuid NOT NULL REFERENCES users(user_id),
    question text NOT NULL,
    answer text,
    query_conditions jsonb NOT NULL DEFAULT '{}'::jsonb,
    stop_reason text,
    status varchar(16) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'stopped')),
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE tool_calls (
    tool_call_id bigserial PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES qa_sessions(session_id) ON DELETE CASCADE,
    step_no integer NOT NULL CHECK (step_no > 0),
    tool_name varchar(32) NOT NULL
        CHECK (tool_name IN ('query_records', 'search_manual', 'get_event_video')),
    input jsonb NOT NULL DEFAULT '{}'::jsonb,
    result_status varchar(20) NOT NULL
        CHECK (result_status IN ('ok', 'empty', 'error', 'not_received', 'rejected')),
    output jsonb,
    reason text,
    rejected_reason text,
    called_at timestamptz NOT NULL DEFAULT now(),
    duration_ms integer CHECK (duration_ms >= 0),
    UNIQUE (session_id, step_no)
);

CREATE TABLE manual_documents (
    document_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_code varchar(32) NOT NULL,
    doc_name varchar(200) NOT NULL,
    doc_version varchar(32) NOT NULL,
    status varchar(16) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_code, doc_version)
);

CREATE TABLE manual_chunks (
    chunk_id varchar(64) PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES manual_documents(document_id),
    section varchar(200),
    text text NOT NULL,
    chunk_order integer NOT NULL CHECK (chunk_order >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_order)
);

CREATE TABLE tool_call_evidence (
    tool_call_id bigint NOT NULL REFERENCES tool_calls(tool_call_id) ON DELETE CASCADE,
    chunk_id varchar(64) NOT NULL REFERENCES manual_chunks(chunk_id),
    similarity numeric(8,6) CHECK (similarity BETWEEN 0 AND 1),
    PRIMARY KEY (tool_call_id, chunk_id)
);

CREATE TABLE system_configs (
    config_key varchar(160) PRIMARY KEY,
    config_value jsonb NOT NULL,
    description text,
    updated_by uuid REFERENCES users(user_id),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit_logs (
    audit_id bigserial PRIMARY KEY,
    store_id varchar(32) REFERENCES stores(store_id),
    actor_type varchar(16) NOT NULL
        CHECK (actor_type IN ('user', 'agent', 'system')),
    actor_id varchar(128),
    action varchar(64) NOT NULL,
    entity_type varchar(64) NOT NULL,
    entity_id varchar(128) NOT NULL,
    before_data jsonb,
    after_data jsonb,
    reason text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ix_users_store ON users (store_id);
CREATE INDEX ix_auth_sessions_expiry ON auth_sessions (expires_at);
CREATE INDEX ix_events_store_occurred ON events (store_id, occurred_at DESC);
CREATE INDEX ix_events_store_status_occurred
    ON events (store_id, status, occurred_at DESC);
CREATE INDEX ix_events_camera_occurred ON events (camera_id, occurred_at DESC);
CREATE INDEX ix_event_history_event_changed
    ON event_status_history (event_id, changed_at DESC);
CREATE INDEX ix_order_drafts_store_target
    ON order_drafts (store_id, target_date DESC);
CREATE INDEX ix_qa_sessions_store_created
    ON qa_sessions (store_id, created_at DESC);
CREATE INDEX ix_tool_calls_session_step
    ON tool_calls (session_id, step_no);

-- 매장·카메라가 다른 사건을 저장하지 않도록 복합 외래키를 추가합니다.
ALTER TABLE cameras
    ADD CONSTRAINT uq_cameras_store_camera UNIQUE (store_id, camera_id);

ALTER TABLE events
    ADD CONSTRAINT fk_events_store_camera
    FOREIGN KEY (store_id, camera_id)
    REFERENCES cameras (store_id, camera_id);

-- 상태 변경 이력의 주체와 대상 매장을 서비스 계층에서 함께 검증합니다.
-- users.store_id와 event.store_id의 교차 검증은 애플리케이션 트랜잭션에서 수행합니다.