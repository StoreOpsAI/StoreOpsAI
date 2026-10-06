import { request } from "./client";

// 현재 로그인 세션으로 매장 기록과 점검 규정을 조회합니다.
export function askQuestion(question) {
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

// 질문을 실행한 도구와 입력, 결과, 선택 이유를 확인합니다.
export function getQuestionLog(sessionId) {
  return request(`/api/ask/${sessionId}/log`);
}
