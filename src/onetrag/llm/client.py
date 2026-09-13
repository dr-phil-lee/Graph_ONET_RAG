"""Single entry point for all Claude API calls: cache-first, retried, logged.

Every experimental stage goes through LLM.complete(), so token usage and cost
accounting live in one place. count_tokens() measures serialized context for
the matched retrieval budget; it uses the API when a key is present and falls
back to a chars/4 estimate offline (flagged in the return value so runs can
report which measurement was used).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from ..config import config_hash, data_path
from .cache import LLMCache


class LLM:
    def __init__(self, cfg: dict, cache_file: str | Path | None = None):
        self.cfg = cfg
        self.cfg_hash = config_hash(cfg)
        self.cache = LLMCache(cache_file or data_path(cfg, "llm_cache.sqlite"))
        self._client = None
        self.totals = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def client(self):
        if self._client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add the key.")
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, *, model: str, system: str, user: str,
                 max_tokens: int | None = None, temperature: float | None = None,
                 use_cache: bool = True) -> dict:
        # `temperature` is removed on the Claude 4.6+ / 5 families (400 error),
        # so it is never sent; the kwarg remains for call-site compatibility.
        params = {
            "model": model, "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens or self.cfg["generation"]["max_tokens"],
        }
        key = LLMCache.make_key({**params, "config": self.cfg_hash})
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.totals["cache_hits"] += 1
                return hit

        delay = 2.0
        for attempt in range(6):
            try:
                t0 = time.time()
                resp = self.client.messages.create(**params)
                text = "".join(b.text for b in resp.content if b.type == "text")
                usage = {"input_tokens": resp.usage.input_tokens,
                         "output_tokens": resp.usage.output_tokens,
                         "latency_s": round(time.time() - t0, 2)}
                self.totals["calls"] += 1
                self.totals["input_tokens"] += usage["input_tokens"]
                self.totals["output_tokens"] += usage["output_tokens"]
                self.cache.put(key, model, text, usage)
                return {"text": text, "usage": usage, "cached": False}
            except Exception as e:  # noqa: BLE001 — retry transient API errors
                import anthropic
                transient = isinstance(e, (anthropic.RateLimitError, anthropic.APIStatusError,
                                           anthropic.APIConnectionError))
                if not transient or attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def count_tokens(self, text: str, model: str | None = None) -> tuple[int, str]:
        """(token count, method). API-accurate with a key, chars/4 estimate without."""
        if os.getenv("ANTHROPIC_API_KEY"):
            key = LLMCache.make_key({"count": text, "model": model or "default"})
            hit = self.cache.get(key)
            if hit is not None:
                return int(hit["text"]), "api"
            resp = self.client.messages.count_tokens(
                model=model or self.cfg["models"]["generator"],
                messages=[{"role": "user", "content": text}])
            n = int(resp.input_tokens)
            self.cache.put(key, "count_tokens", str(n), {})
            return n, "api"
        return max(1, len(text) // 4), "estimate"
