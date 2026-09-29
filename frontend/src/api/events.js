import { request } from "./client";

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

// 사건 영상은 세션 헤더가 필요하므로 인증된 Blob으로 받아 화면에 연결합니다.
function getSourceEventId(eventId, uri) {
  const fileName = uri?.replaceAll("\\", "/").split("/").at(-1) ?? "";
  return fileName.match(/^(E\d+)_/)?.[1] ?? eventId;
}

async function fetchDetectorMedia(path) {
  const response = await fetch(`/detector${path}`);
  if (!response.ok) throw new Error("사건 미디어를 불러오지 못했습니다.");
  return URL.createObjectURL(await response.blob());
}

export function getEventClip(eventId, clipUri) {
  if (/^https?:\/\//i.test(clipUri ?? "")) return Promise.resolve(clipUri);
  const sourceEventId = getSourceEventId(eventId, clipUri);
  return fetchDetectorMedia(`/path1/events/${sourceEventId}/clip`);
}

export function getEventRepresentativeImage(eventId, media) {
  if (/^https?:\/\//i.test(media.uri ?? "")) return Promise.resolve(media.uri);
  const sourceEventId = getSourceEventId(eventId, media.uri);
  const suffix = media.uri
    ?.replaceAll("\\", "/")
    .split("/")
    .at(-1)
    ?.match(/_(\d+)\.[^.]+$/);
  const mediaIndices = { frame_start: 0, frame_middle: 1, frame_end: 2 };
  const index = suffix ? Number(suffix[1]) - 1 : mediaIndices[media.media_type];
  if (!Number.isInteger(index) || index < 0) {
    return Promise.reject(new Error("대표 이미지 순서를 확인할 수 없습니다."));
  }
  return fetchDetectorMedia(`/path1/events/${sourceEventId}/images/${index}`);
}
