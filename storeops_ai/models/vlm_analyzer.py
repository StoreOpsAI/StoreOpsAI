"""FR-EVT-13~16 VLM 어댑터.

OPENAI_API_KEY가 설정되면 OpenAI Responses API를 사용하고, 없으면 미설정 상태를
명시적으로 저장한다. 이 모듈은
- 대표 이미지 3장(first/middle/last) 요청 payload 생성
- 3개 필드 검증
- 비동기 실행
을 구현한다.
"""
from __future__ import annotations
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.config import OPENAI_VLM_MODEL, OPENAI_VLM_TIMEOUT_SEC, VLM_MAX_WORKERS


class VLMAnalyzer:
    REQUIRED_KEYS = ("observation", "uncertain_points", "owner_actions")

    def __init__(self, transport: Optional[Callable[[List[str], Dict[str, Any]], Dict[str, Any]]] = None):
        # 테스트 주입 transport가 없으면 OPENAI_API_KEY를 사용하는 실제 Responses API transport를 쓴다.
        self.transport = transport or (self._openai_transport if os.getenv("OPENAI_API_KEY") else None)
        self.executor = ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS)

    @classmethod
    def build_request(cls, image_paths: List[str], event_context: Dict[str, Any]) -> Dict[str, Any]:
        if len(image_paths) != 3:
            raise ValueError(f"VLM 입력 이미지는 처음/가운데/끝 3장이 필요합니다. 현재 {len(image_paths)}장")
        return {
            "images": {
                "first": image_paths[0],
                "middle": image_paths[1],
                "last": image_paths[2],
            },
            "prompt": {
                "observation": "관찰한 내용을 작성 (확인되지 않은 사실을 단정하지 말 것)",
                "uncertain_points": "확실하지 않은 점을 작성. 없으면 '없음'이라고 작성",
                "owner_actions": "점주가 확인할 일 정확히 3가지를 작성",
            },
            "event_context": event_context,
        }

    def analyze(self, image_paths: List[str], event_context: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.build_request(image_paths, event_context)
        if self.transport is None:
            return {
                "observation": "",
                "uncertain_points": "",
                "owner_actions": [],
                "status": "VLM_NOT_CONFIGURED",
                "request": payload,
            }
        result = self.transport(image_paths, event_context)
        if not self.validate(result):
            raise ValueError("VLM 결과는 비어 있지 않은 observation/uncertain_points 및 정확히 3개 owner_actions를 포함해야 합니다.")
        return result

    @staticmethod
    def _image_data_url(path: str) -> str:
        suffix = path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _openai_transport(self, image_paths: List[str], event_context: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI Responses API에 로컬 대표 이미지 3장을 data URL로 전달한다."""
        instructions = (
            "당신은 매장 CCTV 사건 검토자입니다. 이미지에서 보이는 사실만 한국어로 작성하세요. "
            "반드시 JSON 객체만 반환하세요: {\"observation\": string, \"uncertain_points\": string, "
            "\"owner_actions\": [string, string, string]}. owner_actions는 정확히 세 항목이어야 합니다."
        )
        content = [{"type": "input_text", "text": instructions + "\n사건 정보: " + json.dumps(event_context, ensure_ascii=False)}]
        for path in image_paths:
            content.append({"type": "input_image", "image_url": self._image_data_url(path), "detail": "high"})
        payload = {
            "model": OPENAI_VLM_MODEL,
            "input": [{"role": "user", "content": content}],
        }
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=OPENAI_VLM_TIMEOUT_SEC) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError) as exc:
            raise RuntimeError(f"OpenAI VLM request failed: {exc}") from exc
        text = raw.get("output_text", "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI VLM returned non-JSON output") from exc

    def analyze_async(self, image_paths: List[str], event_context: Dict[str, Any]):
        # FR-EVT-15: 사건 저장/알림과 분리된 작업
        return self.executor.submit(self.analyze, image_paths, event_context)

    @classmethod
    def validate(cls, result: Any) -> bool:
        return (
            isinstance(result, dict)
            and all(isinstance(result.get(key), str) and result[key].strip()
                    for key in ("observation", "uncertain_points"))
            and isinstance(result.get("owner_actions"), list)
            and len(result["owner_actions"]) == 3
            and all(isinstance(action, str) and action.strip() for action in result["owner_actions"])
        )
