import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

// StrictMode로 개발 중 부수 효과 문제를 빠르게 발견하면서 앱을 마운트합니다.
createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
