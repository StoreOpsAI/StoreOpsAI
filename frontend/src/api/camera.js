import { SESSION_HEADER_NAME, getSessionId } from "./client";

// 탐지 PC 카메라의 최신 화면(JPEG)을 점주 세션으로 백엔드를 거쳐 받습니다.
// after보다 새 화면이 생길 때까지 백엔드가 기다렸다 돌려주고, 끝내 없으면 null(204)입니다.
export async function fetchLiveFrame(after) {
  const sessionId = getSessionId();
  const response = await fetch(`/api/camera/live-frame?after=${after}`, {
    headers: sessionId ? { [SESSION_HEADER_NAME]: sessionId } : {},
    cache: "no-store",
  });
  if (response.status === 204) return null;
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "실시간 화면을 받지 못했습니다.");
  }
  return {
    seq: Number(response.headers.get("X-Frame-Seq")) || 0,
    blob: await response.blob(),
  };
}
