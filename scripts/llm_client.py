"""
Lightweight LLM client that supports custom endpoint, key and model.
Tries to be compatible with OpenAI-style chat/completions endpoints and generic prompt endpoints.
"""
import json
import time
from typing import Optional, Dict, Any

import requests


class LLMClient:
    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.headers = headers.copy() if headers else {}
        if api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.headers.setdefault("Content-Type", "application/json")

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint, headers=self.headers, json=payload, timeout=self.timeout
                )
                # raise for status to catch 4xx/5xx
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                sleep_t = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_t)
        # if we get here, all retries failed
        raise last_exc

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text for a single prompt.
        Returns a dict with keys: text (str), usage (dict or None), raw (full response dict)
        Behavior:
        - If the endpoint path looks like a chat completion (contains 'chat' or '/v1/chat'), it will send messages
        - Otherwise sends a single 'prompt' field
        """
        extra = extra or {}
        # build payload
        payload: Dict[str, Any] = {}
        url_lower = self.endpoint.lower()
        if "/v1/chat" in url_lower or "chat" in url_lower:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload.update({"messages": messages})
            if self.model:
                payload["model"] = self.model
        else:
            payload.update({"prompt": prompt})
            if self.model:
                payload["model"] = self.model

        payload.setdefault("temperature", temperature)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)

        resp_json = self._post(payload)

        # extract text from common response shapes
        text = None
        usage = None
        if isinstance(resp_json, dict):
            # OpenAI chat style
            choices = resp_json.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                # chat completions
                if isinstance(first.get("message"), dict):
                    text = first["message"].get("content")
                else:
                    text = first.get("text")
            # usage
            usage = resp_json.get("usage")

        # fallback to raw serialized text
        if text is None:
            try:
                text = json.dumps(resp_json, ensure_ascii=False)
            except Exception:
                text = str(resp_json)

        return {"text": text, "usage": usage, "raw": resp_json}
