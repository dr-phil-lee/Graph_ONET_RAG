"""OpenRouter cloud backend satisfying the onetrag LLM contract.

Lets the user swap in any hosted model (e.g. "anthropic/claude-haiku-4.5",
"qwen/qwen-2.5-72b-instruct"). The API key comes from the UI, the REST API
caller, or Chatbot/.env (OPENROUTER_API_KEY) — never from committed files.

Error policy (matters for an API server):
- 408/429/5xx and connection errors are transient: retried 3x with backoff.
- Any other 4xx (bad key, no credits, unknown model, bad request) is raised
  immediately as OpenRouterError with the provider's own message, so callers
  see "invalid API key" instead of three slow, identical failures.
- A 400 while JSON mode is on is retried once without `response_format`
  (some providers do not support it); the second failure is final.
"""
from __future__ import annotations

import time

import requests

import bootstrap

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Non-retryable error from OpenRouter (carries the HTTP status)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _Transient(RuntimeError):
    pass


def _error_message(r: requests.Response) -> str:
    """OpenRouter error bodies look like {"error": {"message": ..., "code": ...}}."""
    try:
        body = r.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            meta = err.get("metadata") or {}
            raw = meta.get("raw") if isinstance(meta, dict) else None
            msg = str(err["message"])
            if raw and str(raw) not in msg:
                msg += f" ({str(raw)[:200]})"
            return f"HTTP {r.status_code}: {msg}"
        if err:
            return f"HTTP {r.status_code}: {str(err)[:300]}"
    return f"HTTP {r.status_code}: {r.text[:300]}"


class OpenRouterLLM:
    def __init__(self, api_key: str,
                 model: str = bootstrap.OPENROUTER_DEFAULT_MODEL,
                 json_mode: bool = True,
                 timeout_s: float = bootstrap.LLM_TIMEOUT_S):
        if not api_key or not api_key.strip():
            raise ValueError("OpenRouter API key is required")
        self.api_key = api_key.strip()
        self.model = (model or bootstrap.OPENROUTER_DEFAULT_MODEL).strip()
        self.json_mode = json_mode
        self.timeout_s = timeout_s
        self.totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def description(self) -> str:
        return f"OpenRouter: {self.model}"

    # ------------------------------------------------------------------ API
    def _post(self, payload: dict) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "X-Title": bootstrap.OPENROUTER_APP_TITLE}
        return requests.post(bootstrap.OPENROUTER_URL, json=payload,
                             headers=headers, timeout=self.timeout_s)

    def complete(self, *, model: str, system: str, user: str,
                 max_tokens: int | None = None, temperature: float | None = None,
                 use_cache: bool = True) -> dict:
        # `model` from cfg is a label for the research pipeline; the model the
        # caller picked (self.model) wins.
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens or bootstrap.GEN_MAX_TOKENS,
            "temperature": bootstrap.GEN_TEMPERATURE if temperature is None else temperature,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        data: dict | None = None
        for attempt in range(3):
            try:
                r = self._post(payload)
                if r.status_code == 400 and "response_format" in payload:
                    # provider may not support JSON mode; retry once without it
                    payload.pop("response_format", None)
                    r = self._post(payload)
                if r.status_code in _TRANSIENT_STATUS:
                    raise _Transient(_error_message(r))
                if r.status_code >= 400:
                    raise OpenRouterError(_error_message(r), r.status_code)
                try:
                    data = r.json()
                except ValueError:
                    raise _Transient(f"non-JSON response: {r.text[:200]}")
                if not isinstance(data, dict) or "choices" not in data:
                    # OpenRouter can return an error body with HTTP 200
                    err = (data or {}).get("error") if isinstance(data, dict) else None
                    code = err.get("code") if isinstance(err, dict) else None
                    msg = str((err or {}).get("message", data))[:300] if isinstance(err, dict) \
                        else str(data)[:300]
                    if code in _TRANSIENT_STATUS:
                        raise _Transient(msg)
                    raise OpenRouterError(msg, code if isinstance(code, int) else None)
                break
            except (requests.ConnectionError, requests.Timeout, _Transient) as e:
                if attempt == 2:
                    raise OpenRouterError(
                        f"OpenRouter request failed after 3 attempts: {e}") from e
                time.sleep(2 ** attempt)
        assert data is not None
        latency = time.perf_counter() - t0

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if isinstance(text, list):  # some providers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        u = data.get("usage") or {}
        usage = {"input_tokens": int(u.get("prompt_tokens") or 0),
                 "output_tokens": int(u.get("completion_tokens") or 0),
                 "latency_s": round(latency, 3),
                 "finish_reason": choice.get("finish_reason")}
        self.totals["calls"] += 1
        self.totals["input_tokens"] += usage["input_tokens"]
        self.totals["output_tokens"] += usage["output_tokens"]
        return {"text": text, "usage": usage, "cached": False,
                "model": data.get("model") or self.model}

    def count_tokens(self, text: str, model: str | None = None) -> tuple[int, str]:
        return max(1, len(text) // 4), "estimate"


def list_models(timeout_s: float = 10) -> list[str]:
    """Sorted OpenRouter model ids (no key needed). Raises on network error."""
    r = requests.get(bootstrap.OPENROUTER_MODELS_URL, timeout=timeout_s)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []) if m.get("id"))
