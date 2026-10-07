"""FR-EVT-13~16 Qwen3-VL VLM 어댑터.

Qwen3-VL-8B-Instruct를 vLLM 등 OpenAI 호환 서버로 실행한 뒤,
사건 대표 이미지 3장(first/middle/last)을 ``/v1/chat/completions``에 보낸다.
VLM 지연/실패는 사건 생성이나 1차 알림을 막지 않는다.
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from config.config import (
    QWEN_VLM_API_KEY,
    QWEN_VLM_BASE_URL,
    QWEN_VLM_IMAGE_JPEG_QUALITY,
    QWEN_VLM_IMAGE_MAX_PIXELS,
    QWEN_VLM_MAX_TOKENS,
    QWEN_VLM_MODEL,
    QWEN_VLM_TIMEOUT_SEC,
    VLM_MAX_WORKERS,
)


class VLMAnalyzer:
    """Qwen3-VL로 사건 대표 이미지 세 장을 설명한다."""

    REQUIRED_KEYS = ("observation", "uncertain_points", "owner_actions")
    MAX_RETRIES = 3
    RETRY_DELAYS = [3, 6, 12]
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        transport: Optional[Callable[[List[str], Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.transport = transport or (
            self._qwen_transport if QWEN_VLM_BASE_URL else None
        )
        self.executor = ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS)

    @classmethod
    def build_request(
        cls,
        image_paths: List[str],
        event_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if len(image_paths) != 3:
            raise ValueError(
                "VLM 입력 이미지는 처음/가운데/끝 3장이 필요합니다. "
                f"현재 {len(image_paths)}장"
            )
        return {
            "images": {
                "first": image_paths[0],
                "middle": image_paths[1],
                "last": image_paths[2],
            },
            "event_context": {
                key: event_context[key]
                for key in ("event_id", "camera_id")
                if key in event_context
            },
        }

    def analyze(
        self,
        image_paths: List[str],
        event_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_info = self.build_request(image_paths, event_context)
        if self.transport is None:
            return {
                "observation": "",
                "uncertain_points": "",
                "owner_actions": [],
                "status": "VLM_NOT_CONFIGURED",
                "request": request_info,
            }

        result = self._normalize_result(self.transport(image_paths, event_context))
        if not self.validate(result):
            raise ValueError(
                "VLM 결과는 비어 있지 않은 observation/uncertain_points 및 "
                "정확히 3개 owner_actions를 포함해야 합니다.\n"
                "Qwen 응답:\n"
                + json.dumps(result, ensure_ascii=False, indent=2)
            )
        result.setdefault("status", "SUCCESS")
        result.setdefault("model_name", QWEN_VLM_MODEL)
        return result

    @staticmethod
    def _normalize_result(result: Any) -> Any:
        """VLM이 '불확실한 점 없음'을 빈 문자열로 낸 경우만 명시값으로 보정한다."""
        if not isinstance(result, dict):
            return result
        uncertain_points = result.get("uncertain_points")
        if isinstance(uncertain_points, str) and not uncertain_points.strip():
            result = dict(result)
            result["uncertain_points"] = "없음"
        return result

    @staticmethod
    def _image_data_url(path: str) -> str:
        """원본은 보존하고 Qwen 전송용 이미지 하나만 축소·JPEG 인코딩한다."""
        encoded_file = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded_file, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"VLM 입력 이미지를 읽을 수 없습니다: {path}")

        height, width = image.shape[:2]
        pixels = width * height
        if pixels > QWEN_VLM_IMAGE_MAX_PIXELS:
            scale = (QWEN_VLM_IMAGE_MAX_PIXELS / pixels) ** 0.5
            target_width = max(1, round(width * scale))
            target_height = max(1, round(height * scale))
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded_image = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, QWEN_VLM_IMAGE_JPEG_QUALITY],
        )
        if not ok:
            raise ValueError(f"VLM 전송용 JPEG 인코딩에 실패했습니다: {path}")

        encoded = base64.b64encode(encoded_image.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _qwen_transport(
        self,
        image_paths: List[str],
        event_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """OpenAI 호환 vLLM의 Qwen3-VL chat completions 엔드포인트를 호출한다."""
        instructions = (
            "당신은 매장 CCTV 사건 검토자입니다. 이미지에서 실제로 확인되는 "
            "내용만 작성하고, 확인되지 않은 사실은 추측하지 마세요. 반드시 JSON "
            "객체만 반환하세요. JSON은 observation(문자열), uncertain_points(문자열), "
            "owner_actions(점주 확인사항 문자열 정확히 3개 배열)를 가져야 합니다."
        )
        image_parts = [
            {"type": "image_url", "image_url": {"url": self._image_data_url(path)}}
            for path in image_paths
        ]
        payload = {
            "model": QWEN_VLM_MODEL,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "사건 정보:\n"
                            + json.dumps(event_context, ensure_ascii=False),
                        },
                        *image_parts,
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": QWEN_VLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if QWEN_VLM_API_KEY:
            headers["Authorization"] = f"Bearer {QWEN_VLM_API_KEY}"
        request = Request(
            f"{QWEN_VLM_BASE_URL}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                print(f"[Qwen VLM] 요청 {attempt + 1}/{self.MAX_RETRIES + 1}")
                with urlopen(request, timeout=QWEN_VLM_TIMEOUT_SEC) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                print("[Qwen VLM] 응답 성공")
                return self._parse_json(self._extract_response_text(raw))
            except HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8")
                except Exception:
                    error_body = ""
                if exc.code not in self.RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"Qwen VLM HTTP 오류: {exc.code}\n{error_body}"
                    ) from exc
                if attempt >= self.MAX_RETRIES:
                    raise RuntimeError(
                        "Qwen VLM 요청이 3회 재시도 후에도 실패했습니다.\n"
                        f"HTTP {exc.code}\n{error_body}"
                    ) from exc
                self._wait_before_retry(attempt, f"HTTP 오류 {exc.code}")
            except (URLError, OSError) as exc:
                if attempt >= self.MAX_RETRIES:
                    raise RuntimeError(
                        "Qwen VLM 서버 연결이 3회 재시도 후에도 실패했습니다.\n"
                        f"{exc}"
                    ) from exc
                self._wait_before_retry(attempt, f"연결 오류: {exc}")

        raise RuntimeError("Qwen VLM 요청 재시도 로직 오류")

    def _wait_before_retry(self, attempt: int, reason: str) -> None:
        delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
        print(f"[Qwen VLM] {reason}; {delay}초 후 재시도합니다.")
        time.sleep(delay)

    @staticmethod
    def _extract_response_text(raw: Dict[str, Any]) -> str:
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Qwen VLM 응답에 choices[0].message.content가 없습니다."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen VLM 응답에 텍스트가 없습니다.")
        return content

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Qwen VLM이 올바른 JSON을 반환하지 않았습니다.\n응답:\n{text}"
            ) from exc

    def analyze_async(self, image_paths: List[str], event_context: Dict[str, Any]):
        return self.executor.submit(self.analyze, image_paths, event_context)

    @classmethod
    def validate(cls, result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        if not isinstance(result.get("observation"), str) or not result["observation"].strip():
            return False
        if not isinstance(result.get("uncertain_points"), str) or not result["uncertain_points"].strip():
            return False
        owner_actions = result.get("owner_actions")
        return (
            isinstance(owner_actions, list)
            and len(owner_actions) == 3
            and all(isinstance(action, str) and action.strip() for action in owner_actions)
        )

    def close(self):
        self.executor.shutdown(wait=False)
