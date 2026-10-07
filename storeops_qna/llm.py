"""OpenAI 호환 /v1/chat/completions 클라이언트 (llama-server 용). 표준 라이브러리만 씁니다."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import LLMConfig


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_mode: bool = False,
        thinking: bool | None = None,
    ) -> dict[str, Any]:
        """assistant 메시지(dict)를 돌려준다: content, tool_calls, reasoning_content 등."""
        c = self.cfg
        body: dict[str, Any] = {
            "model": c.model,
            "messages": messages,
            "temperature": c.temperature,
            "top_p": c.top_p,
            "top_k": c.top_k,
            "presence_penalty": c.presence_penalty,
            "max_tokens": c.max_tokens,
            "chat_template_kwargs": {"enable_thinking": c.enable_thinking if thinking is None else thinking},
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            c.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {c.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=c.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"LLM 서버 오류 {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"LLM 호출 실패: {e}") from e
