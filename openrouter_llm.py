"""OpenRouter cloud backend satisfying the onetrag LLM contract.

Lets the user swap in any hosted model (e.g. "anthropic/claude-haiku-4.5",
"qwen/qwen-2.5-72b-instruct") for much faster answers than the local CPU
model. The API key comes from the UI or Chatbot/.env (OPENROUTER_API_KEY),
never from committed files.
"""
from __future__ import annotations

import time

import requests

import bootstrap


class OpenRouterLLM:
    def __init__(self, api_key: str,
                 model: str = bootstrap.OPENROUTER_DEFAULT_MODEL,
                 json_mode: bool = True):
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self.api_key = api_key
        self.model = model
        self.json_mode = json_mode
        self.totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def description(self) -> str:
        return f"OpenRouter: {self.model}"

    def complete(self, *, model: str, system: str, user: str,
                 max_tokens: int | None = None, temperature: float | None = None,
                 use_cache: bool = True) -> dict:
        # `model` from cfg is a label for the research pipeline; the model the
        # user picked in the UI (self.model) wins.
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens or bootstrap.GEN_MAX_TOKENS,
            "temperature": bootstrap.GEN_TEMPERATURE if temperature is None else temperature,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.post(bootstrap.OPENROUTER_URL, json=payload,
                                  headers=headers, timeout=120)
                if r.status_code == 400 and "response_format" in payload:
                    # model doesn't support JSON mode; retry once without it
                    payload.pop("response_format", None)
                    r = requests.post(bootstrap.OPENROUTER_URL, json=payload,
                                      headers=headers, timeout=120)
                if r.status_code in (429, 500, 502, 503):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                if "choices" not in data:  # OpenRouter error body with 200
                    raise RuntimeError(str(data.get("error", data))[:300])
                break
            except (requests.RequestException, RuntimeError) as e:
                last_err = e
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        latency = time.perf_counter() - t0

        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage", {})
        usage = {"input_tokens": int(u.get("prompt_tokens", 0)),
                 "output_tokens": int(u.get("completion_tokens", 0)),
                 "latency_s": latency}
        self.totals["calls"] += 1
        self.totals["input_tokens"] += usage["input_tokens"]
        self.totals["output_tokens"] += usage["output_tokens"]
        return {"text": text, "usage": usage, "cached": False}

    def count_tokens(self, text: str, model: str | None = None) -> tuple[int, str]:
        return max(1, len(text) // 4), "estimate"
