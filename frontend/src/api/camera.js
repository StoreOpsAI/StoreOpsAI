import { request } from "./client";

// 웹캠 프레임은 로그인한 점주의 세션으로 백엔드를 거쳐 탐지 서비스에 전달합니다.
export function analyzeCameraBatch(batch) {
  return request("/api/camera/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(batch),
  });
}
