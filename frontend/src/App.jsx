import { useEffect, useState } from "react";
import { getCurrentUser, logIn, logOut, signUp } from "./api/auth";
import {
  getEvent,
  getEventClip,
  getEventRepresentativeImage,
  getEvents,
  updateEventStatus,
} from "./api/events";
import { approveDraft, createForecastDraft, getDrafts } from "./api/demand";
import { askQuestion, getQuestionLog } from "./api/questions";
import "./App.css";

// 서버의 상태 값은 운영 화면에서 읽기 쉬운 한국어로 표시합니다.
const statusLabels = {
  unconfirmed: "미확인",
  confirmed: "확인",
  resolved: "처리 완료",
  false_alarm: "오탐",
};
const eventTypeLabels = {
  camera_disconnect: "카메라 끊김",
  camera_reconnected: "카메라 재연결",
  fall: "쓰러짐 의심",
  fight: "싸움 의심",
  fire: "방화 의심",
  theft: "절도 의심",
  vandalism: "기물 훼손",
  littering: "쓰레기 투기",
  broken: "기물 훼손",
  abandon: "쓰레기 투기",
  카메라끊김: "카메라 끊김",
  전도: "쓰러짐 의심",
  쓰러짐: "쓰러짐 의심",
  싸움: "싸움 의심",
  "쓰레기 투기": "쓰레기 투기",
  파손: "기물 훼손",
  방화: "방화 의심",
  유기: "쓰레기 투기",
  절도: "절도 의심",
  폭행: "폭행 의심",
};
const scoreLabels = {
  normal: "정상",
  fall: "쓰러짐",
  fight: "싸움",
  fire: "방화",
  theft: "절도",
  vandalism: "기물 훼손",
  littering: "쓰레기 투기",
  broken: "기물 훼손",
  abandon: "쓰레기 투기",
  전도: "쓰러짐",
  쓰러짐: "쓰러짐",
  싸움: "싸움",
  "쓰레기 투기": "쓰레기 투기",
  파손: "기물 훼손",
  방화: "방화",
  유기: "쓰레기 투기",
  절도: "절도",
  폭행: "폭행",
};

function formatDate(value) {
  // API의 ISO 시각을 점주 화면의 로케일에 맞춰 표시합니다.
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function StatusBadge({ status }) {
  // 상태별 CSS 클래스는 배지의 의미와 색상을 함께 결정합니다.
  return (
    <span className={`status-badge status-${status}`}>
      {statusLabels[status]}
    </span>
  );
}

function EmptyState({ message }) {
  return (
    <div className="empty-state">
      <span>—</span>
      <p>{message}</p>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="error-state">
      <strong>사건을 불러오지 못했습니다</strong>
      <p>{message}</p>
      <button type="button" className="secondary-button" onClick={onRetry}>
        다시 시도
      </button>
    </div>
  );
}

function EventList({ events, selectedId, onSelect }) {
  // 목록에서는 빠른 비교에 필요한 식별자와 핵심 메타데이터만 보여줍니다.
  return (
    <div className="event-list">
      {events.map((event) => (
        <button
          type="button"
          className={`event-row ${event.event_id === selectedId ? "selected" : ""}`}
          key={event.event_id}
          onClick={() => onSelect(event.event_id)}
        >
          <span className="event-row-id">{event.event_id}</span>
          <span className="event-row-main">
            <strong>
              {eventTypeLabels[event.event_type] || event.event_type}
            </strong>
            <small>{formatDate(event.occurred_at)}</small>
          </span>
          <span className="event-row-meta">{event.camera_id}</span>
          <StatusBadge status={event.status} />
        </button>
      ))}
    </div>
  );
}

function RepresentativeImages({ event }) {
  const [imageResult, setImageResult] = useState({ key: "", images: [] });
  const imageRecords = event.representative_images ?? [];
  const imageSignature = JSON.stringify(imageRecords);
  const imageKey = `${event.event_id}:${imageSignature}`;

  useEffect(() => {
    let active = true;
    const objectUrls = [];
    const dbMedia = JSON.parse(imageSignature || "[]");
    Promise.allSettled(
      dbMedia.map((media) =>
        getEventRepresentativeImage(event.event_id, media),
      ),
    ).then((results) => {
      const loadedImages = results.flatMap((result, index) => {
        if (result.status !== "fulfilled") return [];
        objectUrls.push(result.value);
        return [{ url: result.value, media: dbMedia[index] }];
      });
      if (active) setImageResult({ key: imageKey, images: loadedImages });
      else
        objectUrls
          .filter((url) => url.startsWith("blob:"))
          .forEach((url) => URL.revokeObjectURL(url));
    });
    return () => {
      active = false;
      objectUrls
        .filter((url) => url.startsWith("blob:"))
        .forEach((url) => URL.revokeObjectURL(url));
    };
  }, [event.event_id, imageKey, imageSignature]);

  const images = imageResult.key === imageKey ? imageResult.images : [];
  const frameLabels = {
    frame_start: "시작",
    frame_middle: "가운데",
    frame_end: "끝",
  };
  if (!imageRecords.length) {
    return (
      <p className="representative-empty">저장된 대표 이미지가 없습니다.</p>
    );
  }

  return (
    <div className="representative-grid">
      {images.map(({ url, media }) => (
        <figure key={`${event.event_id}-${media.media_type}`}>
          <img
            src={url}
            alt={`${event.event_id} ${frameLabels[media.media_type]} 프레임`}
          />
          <figcaption>{frameLabels[media.media_type]} 프레임</figcaption>
        </figure>
      ))}
    </div>
  );
}

function EventDetail({ event, isSubmitting, onStatusChange }) {
  // 서버의 허용 상태 전이와 동일한 버튼만 노출해 잘못된 요청을 줄입니다.
  const availableActions = {
    unconfirmed: [
      ["confirmed", "확인 처리"],
      ["false_alarm", "오탐 처리"],
    ],
    confirmed: [["resolved", "처리 완료"]],
    resolved: [],
    false_alarm: [],
  };
  const [clipUrl, setClipUrl] = useState("");
  const clipUri = event.clip?.uri;
  const clipLength = event.clip?.length_sec;
  const isPublicClip = /^https?:\/\//i.test(clipUri ?? "");

  useEffect(() => {
    let active = true;
    let objectUrl = "";
    setClipUrl("");
    if (clipUri && !isPublicClip) {
      getEventClip(event.event_id, clipUri)
        .then((url) => {
          objectUrl = url;
          if (active) setClipUrl(url);
        })
        .catch(() => {});
    }
    return () => {
      active = false;
      if (objectUrl?.startsWith("blob:")) URL.revokeObjectURL(objectUrl);
    };
  }, [event.event_id, clipUri, clipLength, isPublicClip]);

  return (
    <article className="detail-panel">
      <div className="detail-heading">
        <div>
          <p className="section-kicker">선택한 사건</p>
          <h2>{event.event_id}</h2>
        </div>
        <StatusBadge status={event.status} />
      </div>
      <div className="detail-meta">
        <div>
          <span>사건 종류</span>
          <strong>
            {eventTypeLabels[event.event_type] || event.event_type}
          </strong>
        </div>
        <div>
          <span>발생 시각</span>
          <strong>{formatDate(event.occurred_at)}</strong>
        </div>
        <div>
          <span>카메라</span>
          <strong>{event.camera_id}</strong>
        </div>
        <div>
          <span>발생 경로</span>
          <strong>
            {event.source === "behavior_model" ? "행동 분류 모델" : "시간 규칙"}
          </strong>
        </div>
      </div>
      <section className="media-panel">
        {isPublicClip || clipUrl ? (
          <video
            className="event-video"
            controls
            preload="metadata"
            src={isPublicClip ? clipUri : clipUrl}
          />
        ) : (
          <div className="media-icon">{event.clip ? "▶" : "∅"}</div>
        )}
        <div>
          <strong>{event.clip ? "사건 영상 준비됨" : "사건 영상 없음"}</strong>
          <p>
            {event.clip
              ? `${event.clip.length_sec}초 클립`
              : "카메라 끊김 사건으로 저장된 영상이 없습니다."}
          </p>
        </div>
      </section>
      {Object.keys(event.scores).length > 0 && (
        <section className="score-section">
          <div className="section-title-row">
            <div>
              <p className="section-kicker">행동 분류</p>
              <h3>모델 점수</h3>
            </div>
            <span className="info-note">사고 확률이 아닌 상대 점수</span>
          </div>
          <div className="score-list">
            {Object.entries(event.scores).map(([key, score]) => (
              <div className="score-item" key={key}>
                <div>
                  <span>{scoreLabels[key] || key}</span>
                  <strong>{Math.round(score * 100)}%</strong>
                </div>
                <div className="score-track">
                  <span style={{ width: `${score * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
          {event.threshold !== null && (
            <p className="threshold">
              기준값 {Math.round(event.threshold * 100)}%
            </p>
          )}
          <p className="threshold">
            현재 저장소의 학습 가중치에는 싸움 클래스가 없어 점수는 0입니다.
          </p>
        </section>
      )}
      <section className="vlm-section">
        <div className="section-title-row">
          <div>
            <p className="section-kicker">참고 정보</p>
            <h3>VLM 설명</h3>
          </div>
          <span className={`vlm-status vlm-${event.vlm.status}`}>
            {event.vlm.status === "pending"
              ? "준비 중"
              : event.vlm.status === "completed"
                ? "완료"
                : event.vlm.status === "not_applicable"
                  ? "대상 아님"
                  : "실패"}
          </span>
        </div>
        <RepresentativeImages event={event} />
        {event.vlm.model_name && (
          <p className="vlm-model">설명 모델 · {event.vlm.model_name}</p>
        )}
        {event.vlm.status === "not_applicable" ? (
          <p>카메라 상태 사건에는 영상 분석을 적용하지 않습니다.</p>
        ) : event.vlm.status === "pending" ? (
          <p className="vlm-pending-message" role="status">
            vLLM 서버에서 이미지 설명을 생성하고 있습니다.
          </p>
        ) : event.vlm.status === "failed" ? (
          <p className="vlm-error">
            {event.vlm.error || "VLM 설명을 생성하지 못했습니다."}
          </p>
        ) : (
          <div className="vlm-result">
            <p>
              <strong>관찰</strong>
              {event.vlm.summary}
            </p>
            <p>
              <strong>불확실한 점</strong>
              {event.vlm.uncertain_points}
            </p>
            <div>
              <strong>점주 확인</strong>
              <ol>
                {event.vlm.owner_actions.map((action, index) => (
                  <li key={`${event.event_id}-vlm-${index}`}>{action}</li>
                ))}
              </ol>
            </div>
          </div>
        )}
        <small>VLM 설명은 행동 분류 판단의 근거가 아닌 참고 정보입니다.</small>
      </section>
      <div className="action-area">
        <div>
          <p className="section-kicker">점주 확인</p>
          <h3>사건 상태 변경</h3>
        </div>
        <div className="action-buttons">
          {availableActions[event.status].map(([nextStatus, label]) => (
            <button
              type="button"
              className={
                nextStatus === "false_alarm"
                  ? "secondary-button"
                  : "primary-button"
              }
              key={nextStatus}
              disabled={isSubmitting}
              onClick={() => onStatusChange(nextStatus)}
            >
              {isSubmitting ? "처리 중..." : label}
            </button>
          ))}
          {availableActions[event.status].length === 0 && (
            <span className="completed-note">
              추가로 변경할 수 없는 상태입니다.
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({
    store_name: "",
    email: "",
    password: "",
    display_name: "",
  });
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (event) => {
    setForm((current) => ({
      ...current,
      [event.target.name]: event.target.value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      const data =
        mode === "login"
          ? await logIn({ email: form.email, password: form.password })
          : await signUp(form).then(() =>
              logIn({ email: form.email, password: form.password }),
            );
      onAuthenticated(data.user);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="auth-shell">
      <section className="auth-panel">
        <div className="auth-brand">
          <span>SO</span>
          <div>
            <strong>StoreOps AI</strong>
            <small>매장 운영 관제</small>
          </div>
        </div>
        <p className="eyebrow">점주 전용 업무 화면</p>
        <h1>{mode === "login" ? "다시 오셨군요" : "매장 운영을 시작하세요"}</h1>
        <p className="auth-description">
          {mode === "login"
            ? "등록한 계정으로 로그인해 매장 사건을 확인합니다."
            : "점주 계정을 만들면 매장별 운영 데이터를 안전하게 관리할 수 있습니다."}
        </p>
        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === "signup" && (
            <>
              <label htmlFor="store_name">매장 이름</label>
              <input
                id="store_name"
                name="store_name"
                value={form.store_name}
                onChange={handleChange}
                required
                placeholder="예: 강남역점"
              />
              <label htmlFor="display_name">점주 이름</label>
              <input
                id="display_name"
                name="display_name"
                value={form.display_name}
                onChange={handleChange}
                required
                placeholder="예: 홍길동"
              />
            </>
          )}
          <label htmlFor="email">이메일</label>
          <input
            id="email"
            name="email"
            type="email"
            value={form.email}
            onChange={handleChange}
            required
            autoComplete="email"
            placeholder="owner@example.com"
          />
          <label htmlFor="password">비밀번호</label>
          <input
            id="password"
            name="password"
            type="password"
            value={form.password}
            onChange={handleChange}
            required
            minLength={mode === "signup" ? 8 : 1}
            autoComplete={
              mode === "login" ? "current-password" : "new-password"
            }
            placeholder={mode === "signup" ? "8자 이상" : "비밀번호 입력"}
          />
          {error && <p className="auth-error">{error}</p>}
          <button
            type="submit"
            className="primary-button auth-submit"
            disabled={isSubmitting}
          >
            {isSubmitting
              ? "처리 중..."
              : mode === "login"
                ? "로그인"
                : "회원가입"}
          </button>
        </form>
        <button
          type="button"
          className="auth-switch"
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setError("");
          }}
        >
          {mode === "login"
            ? "처음 방문하셨나요? 회원가입"
            : "이미 계정이 있나요? 로그인"}
        </button>
      </section>
    </main>
  );
}

function localDateString(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function offsetDate(value, days) {
  const date = new Date(`${value}T00:00:00`);
  date.setDate(date.getDate() + days);
  return localDateString(date);
}

function createInitialOrderForm() {
  const yesterday = offsetDate(localDateString(new Date()), -1);
  return {
    product_id: "P001",
    as_of: yesterday,
    history: Array.from({ length: 7 }, () => ({
      sales: "",
      status: "observed",
    })),
    on_hand: "15",
    incoming: "0",
    holiday_1: false,
    holiday_2: false,
  };
}

function OrderPanel() {
  // 수요 서비스에 전달할 발주 입력과 저장된 초안 목록을 관리합니다.
  const [form, setForm] = useState(createInitialOrderForm);
  const [drafts, setDrafts] = useState([]);
  const [editedQuantities, setEditedQuantities] = useState({});
  const [loadState, setLoadState] = useState("loading");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const orderErrorMessage = (requestError) =>
    /Bearer token required|수요 서비스 인증이 설정되지 않았습니다/i.test(
      requestError.message,
    )
      ? "발주 서비스 인증 설정이 필요합니다. 관리자에게 문의해 주세요."
      : requestError.message;

  const loadDrafts = async () => {
    setLoadState("loading");
    setError("");
    try {
      const loadedDrafts = await getDrafts();
      setDrafts(loadedDrafts);
      setEditedQuantities((current) => {
        const next = { ...current };
        for (const draft of loadedDrafts) {
          if (next[draft.draft_id] === undefined) {
            next[draft.draft_id] = String(draft.recommended.qty);
          }
        }
        return next;
      });
      setLoadState("success");
    } catch (requestError) {
      setError(orderErrorMessage(requestError));
      setLoadState("error");
    }
  };

  useEffect(() => {
    loadDrafts();
  }, []);

  const handleChange = (event) => {
    setForm((current) => ({
      ...current,
      [event.target.name]: event.target.value,
    }));
  };

  const handleHistoryChange = (index, field, value) => {
    setForm((current) => ({
      ...current,
      history: current.history.map((day, dayIndex) =>
        dayIndex === index ? { ...day, [field]: value } : day,
      ),
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError("");
    try {
      await createForecastDraft({
        demand: {
          product_id: form.product_id,
          as_of: form.as_of,
          country: "US",
          history: form.history.map((day, index) => ({
            date: offsetDate(form.as_of, index - 6),
            sales: day.status === "observed" ? Number(day.sales) : null,
            status: day.status,
          })),
          calendar: [1, 2].map((day) => ({
            date: offsetDate(form.as_of, day),
            is_holiday: Number(form[`holiday_${day}`]),
            has_event: 0,
          })),
        },
        on_hand: Number(form.on_hand),
        incoming: Number(form.incoming),
      });
      await loadDrafts();
    } catch (requestError) {
      setError(orderErrorMessage(requestError));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleApprove = async (draft) => {
    const qty = Number(
      editedQuantities[draft.draft_id] ?? draft.recommended.qty,
    );
    if (!Number.isInteger(qty) || qty < 0 || qty % draft.pack_size !== 0) {
      setError(`승인 수량은 ${draft.pack_size}개 단위로 입력해 주세요.`);
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      await approveDraft(draft.draft_id, qty);
      await loadDrafts();
    } catch (requestError) {
      setError(orderErrorMessage(requestError));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="orders-layout">
      <section className="order-form-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-label">MANUAL DEMAND INPUT</span>
            <h2>발주 초안 만들기</h2>
          </div>
        </div>
        <p className="order-form-note">
          미국 M5 공개 판매 자료로 학습한 XGBoost 예측입니다. 최근 7일의 상태를
          구분하며, 휴무·품절·기록 누락을 0으로 대체하지 않습니다. 추천 수량은
          예측 합계 + 안전재고 - 현재 재고 - 입고 예정으로 계산합니다.
        </p>
        <form className="order-form" onSubmit={handleSubmit}>
          <label>
            상품 코드
            <input
              name="product_id"
              value={form.product_id}
              onChange={handleChange}
              required
            />
          </label>
          <label>
            최근 판매 기준일
            <input
              name="as_of"
              type="date"
              value={form.as_of}
              onChange={handleChange}
              required
            />
          </label>
          <fieldset className="history-fields">
            <legend>최근 7일 판매 기록</legend>
            {form.history.map((day, index) => (
              <div className="history-row" key={index}>
                <span>{offsetDate(form.as_of, index - 6)}</span>
                <select
                  aria-label={`${offsetDate(form.as_of, index - 6)} 기록 상태`}
                  value={day.status}
                  onChange={(event) =>
                    handleHistoryChange(index, "status", event.target.value)
                  }
                >
                  <option value="observed">판매 기록 있음</option>
                  <option value="closed">휴무</option>
                  <option value="stockout">품절</option>
                  <option value="missing">기록 누락</option>
                </select>
                <input
                  aria-label={`${offsetDate(form.as_of, index - 6)} 판매량`}
                  type="number"
                  min="0"
                  step="1"
                  placeholder="판매량"
                  value={day.sales}
                  disabled={day.status !== "observed"}
                  required={day.status === "observed"}
                  onChange={(event) =>
                    handleHistoryChange(index, "sales", event.target.value)
                  }
                />
              </div>
            ))}
          </fieldset>
          <fieldset className="forecast-calendar">
            <legend>예측 대상일 달력</legend>
            {[1, 2].map((day) => (
              <label key={day}>
                {day === 1 ? "내일" : "모레"} · {offsetDate(form.as_of, day)}{" "}
                공휴일
                <input
                  type="checkbox"
                  checked={form[`holiday_${day}`]}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      [`holiday_${day}`]: event.target.checked,
                    }))
                  }
                />
              </label>
            ))}
          </fieldset>
          <div className="order-form-row">
            <label>
              현재 재고
              <input
                name="on_hand"
                type="number"
                min="0"
                value={form.on_hand}
                onChange={handleChange}
                required
              />
            </label>
            <label>
              입고 예정
              <input
                name="incoming"
                type="number"
                min="0"
                value={form.incoming}
                onChange={handleChange}
                required
              />
            </label>
          </div>
          <button
            type="submit"
            className="primary-button"
            disabled={isSubmitting}
          >
            {isSubmitting ? "계산 중..." : "추천량 계산"}
          </button>
        </form>
        {error && <p className="inline-error">{error}</p>}
      </section>
      <section className="draft-list-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-label">ORDER DRAFTS</span>
            <h2>
              발주 초안 <em>{drafts.length}</em>
            </h2>
          </div>
          <button type="button" className="text-button" onClick={loadDrafts}>
            새로고침
          </button>
        </div>
        {loadState === "loading" && (
          <div className="loading-state">발주 초안을 불러오는 중입니다...</div>
        )}
        {loadState === "error" && (
          <ErrorState message={error} onRetry={loadDrafts} />
        )}
        {loadState === "success" && drafts.length === 0 && (
          <EmptyState message="아직 발주 초안이 없습니다." />
        )}
        {loadState === "success" && drafts.length > 0 && (
          <div className="draft-list">
            {drafts.map((draft) => (
              <article className="draft-row" key={draft.draft_id}>
                <header className="draft-title">
                  <div>
                    <strong>{draft.product_id}</strong>
                    <small>대상일 {draft.target_date}</small>
                  </div>
                  <div className="draft-tags">
                    <span className={`data-label data-${draft.data_label}`}>
                      {draft.forecast_source === "xgboost"
                        ? "M5 · XGBoost"
                        : draft.data_label === "synthetic"
                          ? "합성·모의 자료"
                          : "실제 자료"}
                    </span>
                    <span className={`draft-status status-${draft.status}`}>
                      {draft.approved ? "승인 완료" : "승인 대기"}
                    </span>
                  </div>
                </header>
                <div className="draft-metrics">
                  <div>
                    <span>내일 수요</span>
                    <strong>{draft.forecast.d1}</strong>
                  </div>
                  <div>
                    <span>모레 수요</span>
                    <strong>{draft.forecast.d2}</strong>
                  </div>
                  <div>
                    <span>총 예상 수요</span>
                    <strong>{draft.forecast.total}</strong>
                  </div>
                  <div>
                    <span>현재 재고</span>
                    <strong>{draft.on_hand}</strong>
                  </div>
                  <div>
                    <span>입고 예정</span>
                    <strong>{draft.incoming}</strong>
                  </div>
                  <div>
                    <span>도착 전 부족량</span>
                    <strong>{draft.shortage_before_arrival}</strong>
                  </div>
                </div>
                <div className="draft-approval-row">
                  <div className="recommended-input">
                    <label htmlFor={`qty-${draft.draft_id}`}>승인 수량</label>
                    <input
                      id={`qty-${draft.draft_id}`}
                      type="number"
                      min="0"
                      step={draft.pack_size}
                      value={
                        draft.approved
                          ? draft.approved.qty
                          : (editedQuantities[draft.draft_id] ??
                            draft.recommended.qty)
                      }
                      disabled={Boolean(draft.approved) || isSubmitting}
                      onChange={(event) =>
                        setEditedQuantities((current) => ({
                          ...current,
                          [draft.draft_id]: event.target.value,
                        }))
                      }
                    />
                    <span>
                      개 ·{" "}
                      {Math.floor(
                        Number(
                          editedQuantities[draft.draft_id] ??
                            draft.recommended.qty,
                        ) / draft.pack_size,
                      )}
                      박스 (1박스 {draft.pack_size}개)
                    </span>
                  </div>
                  {draft.approved ? (
                    <p className="approval-meta">
                      {draft.approved.by} ·{" "}
                      {new Intl.DateTimeFormat("ko-KR", {
                        dateStyle: "medium",
                        timeStyle: "short",
                      }).format(new Date(draft.approved.at))}
                    </p>
                  ) : (
                    <button
                      type="button"
                      className="primary-button"
                      disabled={isSubmitting}
                      onClick={() => handleApprove(draft)}
                    >
                      {isSubmitting ? "승인 중..." : "수량 승인"}
                    </button>
                  )}
                </div>
                <p className="order-safety-note">
                  승인해도 공급사로 주문이 전송되지 않습니다. 추천 수량은 품절
                  방지를 보장하지 않습니다.
                </p>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function QuestionVideo({ source }) {
  const [video, setVideo] = useState({ url: "", error: "" });

  useEffect(() => {
    let active = true;
    let objectUrl = "";
    getEventClip(source.event_id, source.video_uri)
      .then((url) => {
        objectUrl = url;
        if (active) setVideo({ url, error: "" });
      })
      .catch(() => {
        if (active) setVideo({ url: "", error: "영상을 불러오지 못했습니다." });
      });
    return () => {
      active = false;
      if (objectUrl.startsWith("blob:")) URL.revokeObjectURL(objectUrl);
    };
  }, [source.event_id, source.video_uri]);

  if (video.error) return <p className="inline-error">{video.error}</p>;
  return video.url ? (
    <video
      className="question-video"
      src={video.url}
      controls
      aria-label={`${source.event_id} 사건 영상`}
    />
  ) : (
    <p className="loading-state">영상 불러오는 중...</p>
  );
}

function QuestionsPanel() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [log, setLog] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLogLoading, setIsLogLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!question.trim() || isLoading) return;
    setIsLoading(true);
    setError("");
    setLog(null);
    setResult(null);
    try {
      setResult(await askQuestion(question.trim()));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLog = async () => {
    if (log) {
      setLog(null);
      return;
    }
    setIsLogLoading(true);
    try {
      setLog(await getQuestionLog(result.session_id));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLogLoading(false);
    }
  };

  return (
    <div className="questions-layout">
      <form className="question-form" onSubmit={handleSubmit}>
        <label htmlFor="owner-question">매장 운영 질문</label>
        <div className="question-input-row">
          <input
            id="owner-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="매장 기록 또는 점검 규정에 대해 질문하세요"
            maxLength={500}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="primary-button"
            disabled={!question.trim() || isLoading}
          >
            {isLoading ? "조회 중..." : "질문하기"}
          </button>
        </div>
      </form>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {result && (
        <section className="question-result" aria-live="polite">
          <div className="question-result-heading">
            <h2>답변</h2>
            <span>
              질문 #{result.session_id} · 도구 {result.tool_call_count}회
            </span>
          </div>
          <p className="question-answer">{result.answer}</p>
          {result.needs_confirmation && (
            <p className="question-confirmation">
              번호나 조회 조건을 확인한 뒤 다시 질문해 주세요.
            </p>
          )}
          {result.notices
            ?.filter((notice) => notice !== result.answer)
            .map((notice, index) => (
              <p className="question-notice" key={`${index}-${notice}`}>
                {notice}
              </p>
            ))}
          {result.conditions?.length > 0 && (
            <div className="question-details">
              <h3>조회 조건</h3>
              <ul>
                {result.conditions.map((condition, index) => (
                  <li key={`${index}-${condition}`}>{condition}</li>
                ))}
              </ul>
            </div>
          )}
          {result.grounded_sentences?.length > 0 && (
            <div className="question-details">
              <h3>문장별 근거</h3>
              <ol className="question-evidence">
                {result.grounded_sentences.map((sentence, index) => (
                  <li key={`${index}-${sentence.text}`}>
                    <strong>{sentence.text}</strong>
                    {sentence.evidence?.map((evidence) => (
                      <blockquote key={evidence.chunk_id}>
                        <small>
                          {evidence.doc_name} · {evidence.doc_version} ·{" "}
                          {evidence.doc_status || "상태 미지정"} ·{" "}
                          {evidence.section} · {evidence.chunk_id}
                        </small>
                        <p>{evidence.text}</p>
                      </blockquote>
                    ))}
                  </li>
                ))}
              </ol>
            </div>
          )}
          {result.sources
            ?.filter((source) => source.type === "video")
            .map((source) => (
              <div className="question-details" key={source.event_id}>
                <h3>
                  {source.event_id} · {source.camera_id} ·{" "}
                  {source.clip?.length_sec ?? "?"}초
                </h3>
                <QuestionVideo source={source} />
              </div>
            ))}
          <button
            type="button"
            className="question-log-toggle"
            onClick={handleLog}
            disabled={isLogLoading}
          >
            {isLogLoading
              ? "기록 불러오는 중..."
              : log
                ? "실행 기록 닫기"
                : "실행 기록 보기"}
          </button>
          {log && (
            <ol className="question-log">
              {log.tool_calls.map((step) => (
                <li key={step.step}>
                  <strong>
                    {step.tool_name || "모델"} · {step.result_status}
                  </strong>
                  <p>선택 이유: {step.reason || "기록 없음"}</p>
                  <details>
                    <summary>입력과 결과</summary>
                    <pre>
                      {step.input}
                      {"\n"}
                      {step.output || step.rejected_reason}
                    </pre>
                  </details>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}
    </div>
  );
}

function App() {
  const [authState, setAuthState] = useState("loading");
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState("events");
  const [health, setHealth] = useState("연결 확인 중");
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [loadState, setLoadState] = useState("loading");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data) =>
        setHealth(data.status === "ok" ? "API 연결됨" : "API 확인 필요"),
      )
      .catch(() => setHealth("API 연결 실패"));
  }, []);

  useEffect(() => {
    getCurrentUser()
      .then((currentUser) => {
        setUser(currentUser);
        setAuthState("authenticated");
      })
      .catch(() => setAuthState("unauthenticated"));
  }, []);

  const loadEvents = async ({ background = false } = {}) => {
    // 목록을 새로 읽은 뒤 첫 번째 사건을 기본 선택 상태로 맞춥니다.
    if (!background) {
      setLoadState("loading");
      setError("");
    }
    try {
      const data = await getEvents();
      setEvents(data.events);
      setLoadState(data.events.length ? "success" : "empty");
      setError("");
      if (background) {
        setSelectedEvent(
          (currentEvent) =>
            data.events.find(
              (event) => event.event_id === currentEvent?.event_id,
            ) ??
            data.events[0] ??
            null,
        );
      } else {
        setSelectedEvent(
          data.events[0] ? await getEvent(data.events[0].event_id) : null,
        );
      }
    } catch (requestError) {
      if (!background) {
        setError(requestError.message);
        setLoadState("error");
      }
    }
  };

  useEffect(() => {
    if (authState === "authenticated") loadEvents();
  }, [authState]);

  useEffect(() => {
    if (authState !== "authenticated") return undefined;
    const refreshTimer = window.setInterval(
      () => loadEvents({ background: true }),
      5000,
    );
    return () => window.clearInterval(refreshTimer);
  }, [authState]);

  if (authState === "loading")
    return <div className="auth-loading">세션을 확인하는 중입니다...</div>;
  if (authState === "unauthenticated")
    return (
      <AuthScreen
        onAuthenticated={(authenticatedUser) => {
          setUser(authenticatedUser);
          setAuthState("authenticated");
        }}
      />
    );

  const handleLogout = async () => {
    await logOut().catch(() => {});
    setUser(null);
    setAuthState("unauthenticated");
  };

  const handleSelect = async (eventId) => {
    // 목록의 요약 정보 대신 서버의 최신 상세 데이터를 화면에 반영합니다.
    try {
      setSelectedEvent(await getEvent(eventId));
    } catch (requestError) {
      setError(requestError.message);
    }
  };

  const handleStatusChange = async (nextStatus) => {
    // 중복 제출을 막고, 성공 응답을 목록과 상세 화면에 동시에 반영합니다.
    setIsSubmitting(true);
    try {
      const updatedEvent = await updateEventStatus(
        selectedEvent.event_id,
        nextStatus,
      );
      setSelectedEvent(updatedEvent);
      setEvents((currentEvents) =>
        currentEvents.map((event) =>
          event.event_id === updatedEvent.event_id ? updatedEvent : event,
        ),
      );
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">
          <span>SO</span>
          <div>
            <strong>StoreOps AI</strong>
            <small>매장 운영 관제</small>
          </div>
        </div>
        <div className="store-context">
          <span className="live-dot" /> {user.store_id} · {user.store_name}{" "}
          <span className="connection">{health}</span>
          <button
            type="button"
            className="logout-button"
            onClick={handleLogout}
          >
            로그아웃
          </button>
        </div>
      </header>
      <nav className="main-nav" aria-label="주요 업무">
        <button
          className={activeTab === "events" ? "active" : ""}
          onClick={() => setActiveTab("events")}
          type="button"
        >
          <span>01</span> 사건
        </button>
        <button
          className={activeTab === "orders" ? "active" : ""}
          onClick={() => setActiveTab("orders")}
          type="button"
        >
          <span>02</span> 발주
        </button>
        <button
          className={activeTab === "questions" ? "active" : ""}
          onClick={() => setActiveTab("questions")}
          type="button"
        >
          <span>03</span> 질문
        </button>
      </nav>
      <section className="page-content">
        <div className="page-heading">
          <div>
            <p className="eyebrow">
              {activeTab === "questions"
                ? "매장 기록과 규정"
                : "오늘의 운영 현황"}
            </p>
            <h1>
              {activeTab === "orders"
                ? "발주 관리"
                : activeTab === "questions"
                  ? "매장 질문"
                  : "사건 확인"}
            </h1>
            <p>
              {activeTab === "orders"
                ? "수요와 재고를 바탕으로 발주 초안을 만들고 승인합니다."
                : activeTab === "questions"
                  ? "매장 기록을 조회하고 점검 규정의 근거를 확인합니다."
                  : "매장에서 감지된 이상 상황을 검토하고 처리합니다."}
            </p>
          </div>
          {activeTab === "events" && (
            <button
              type="button"
              className="refresh-button"
              onClick={loadEvents}
              disabled={loadState === "loading"}
            >
              ↻ 새로고침
            </button>
          )}
        </div>
        {activeTab === "orders" ? (
          <OrderPanel />
        ) : activeTab === "questions" ? (
          <QuestionsPanel />
        ) : (
          <div className="workspace-grid">
            <section className="list-panel">
              <div className="panel-heading">
                <div>
                  <span className="panel-label">EVENT INBOX</span>
                  <h2>
                    최근 사건 <em>{events.length}</em>
                  </h2>
                </div>
                <span className="unread-count">
                  {
                    events.filter((event) => event.status === "unconfirmed")
                      .length
                  }{" "}
                  미확인
                </span>
              </div>
              {loadState === "loading" && (
                <div className="loading-state">
                  사건 목록을 불러오는 중입니다...
                </div>
              )}
              {loadState === "error" && (
                <ErrorState message={error} onRetry={loadEvents} />
              )}
              {loadState === "empty" && (
                <EmptyState message="표시할 사건이 없습니다." />
              )}
              {loadState === "success" && (
                <EventList
                  events={events}
                  selectedId={selectedEvent?.event_id}
                  onSelect={handleSelect}
                />
              )}
            </section>
            {selectedEvent && (
              <EventDetail
                event={selectedEvent}
                isSubmitting={isSubmitting}
                onStatusChange={handleStatusChange}
              />
            )}
          </div>
        )}
        {error && loadState !== "error" && (
          <p className="inline-error">{error}</p>
        )}
      </section>
    </main>
  );
}

export default App;
