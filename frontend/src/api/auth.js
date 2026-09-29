import { clearSessionId, request, setSessionId } from "./client";

export function getCurrentUser() {
  return request("/api/auth/me");
}

export function signUp(payload) {
  return request("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function logIn(payload) {
  const data = await request("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  setSessionId(data.session_id);
  return data;
}

export async function logOut() {
  try {
    await request("/api/auth/logout", { method: "POST" });
  } finally {
    clearSessionId();
  }
}
