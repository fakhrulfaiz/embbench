from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from embbench.generation.config import Settings, get_settings

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMClient(Protocol):
    def complete_json(self, prompt: str) -> dict[str, Any]: ...


class HttpChatLLM:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, prompt: str) -> str:
        """Plain chat completion. vLLM-friendly; no JSON mode."""
        return self._chat(
            prompt,
            system=None,
            json_mode=False,
        )

    def complete_json(self, prompt: str) -> dict[str, Any]:
        content = self._chat(
            prompt,
            system="You return only valid JSON objects. No markdown fences.",
            json_mode=True,
        )
        return parse_json_object(content)

    def _chat(
        self,
        prompt: str,
        *,
        system: str | None,
        json_mode: bool,
    ) -> str:
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "temperature": 0.4,
            "messages": messages,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self.settings.llm_timeout_s) as client:
            try:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if json_mode and exc.response.status_code in {400, 422}:
                    payload.pop("response_format", None)
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                else:
                    raise
        data = response.json()
        return str(data["choices"][0]["message"]["content"] or "")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON was not an object")
    return obj
