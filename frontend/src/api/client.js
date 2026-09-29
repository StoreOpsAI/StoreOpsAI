const SESSION_STORAGE_KEY = "storeops_session_id";
const SESSION_HEADER_NAME = "X-Session-ID";

// 세션 ID는 로그인 탭 안에서만 유지되도록 sessionStorage에 보관합니다.
export function getSessionId() {
  return sessionStorage.getItem(SESSION_STORAGE_KEY);
}

export function setSessionId(sessionId) {
  sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

export function clearSessionId() {
  sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

// 모든 API 요청의 응답 파싱과 HTTP 오류 변환을 한 곳에서 처리합니다.
export async function request(path, options = {}) {
  const sessionId = getSessionId();
  const headers = { ...options.headers };
  if (sessionId) {
    headers[SESSION_HEADER_NAME] = sessionId;
  }
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "요청을 처리하지 못했습니다.");
  }
  return data;
}
