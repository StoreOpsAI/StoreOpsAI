import { getSessionId, request } from "./client";

// 사건 목록은 현재 선택된 매장의 기본 조건으로 조회합니다.
export function getEvents() {
  return request("/api/events");
}

// 목록에서 선택한 사건의 최신 상세 정보를 조회합니다.
export function getEvent(eventId) {
  return request(`/api/events/${eventId}`);
}

// 점주가 선택한 상태를 서버의 상태 전이 API에 전달합니다.
export function updateEventStatus(eventId, status) {
  return request(`/api/events/${eventId}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, changed_by: "store-owner" }),
  });
}

// 사건 미디어는 백엔드의 점주 세션 인증을 거쳐 받아 화면에 연결합니다.
async function fetchEventMedia(path) {
  const response = await fetch(path, {
    headers: { "X-Session-ID": getSessionId() ?? "" },
  });
  if (!response.ok) throw new Error("사건 미디어를 불러오지 못했습니다.");
  return URL.createObjectURL(await response.blob());
}

export function getEventClip(eventId, clipUri) {
  if (/^https?:\/\//i.test(clipUri ?? "")) return Promise.resolve(clipUri);
  return fetchEventMedia(`/api/events/${encodeURIComponent(eventId)}/clip`);
}

export function getEventRepresentativeImage(eventId, media) {
  if (/^https?:\/\//i.test(media.uri ?? "")) return Promise.resolve(media.uri);
  return fetchEventMedia(
    `/api/events/${encodeURIComponent(eventId)}/representative-images/${media.media_type}`,
  );
}
