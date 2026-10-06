import { request } from "./client";

// 로그인한 점주의 세션을 사용해 수요 서비스의 발주 초안을 조회합니다.
export function getDrafts() {
  return request("/api/demand/orders/drafts");
}

// 입력한 이틀 수요와 재고를 하위 서비스의 발주 계산으로 전달합니다.
export function createDraft(payload) {
  return request("/api/demand/orders/drafts", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": crypto.randomUUID(),
    },
    body: JSON.stringify(payload),
  });
}

// 최근 판매 이력과 달력 정보로 M5 XGBoost 예측 발주 초안을 생성합니다.
export function createForecastDraft(payload) {
  return request("/api/demand/orders/forecast-draft", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": crypto.randomUUID(),
    },
    body: JSON.stringify(payload),
  });
}

// 점주가 계산된 박스 수량을 승인합니다.
export function approveDraft(draftId, qty) {
  return request(`/api/demand/orders/drafts/${draftId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ qty }),
  });
}
