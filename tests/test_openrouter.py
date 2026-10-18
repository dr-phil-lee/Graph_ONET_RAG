"""OpenRouterLLM error policy, with `requests.post` stubbed (no network)."""
from __future__ import annotations

import copy
import json

import pytest

import openrouter_llm as orl
from openrouter_llm import OpenRouterError, OpenRouterLLM


class Resp:
    def __init__(self, status, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _ok(content="{}", model="m"):
    return Resp(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": model})


@pytest.fixture
def posts(monkeypatch):
    """Queue of responses; records payloads. Sleeps are skipped."""
    calls: list[dict] = []
    queue: list[Resp] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": copy.deepcopy(json), "headers": headers})
        return queue.pop(0)

    monkeypatch.setattr(orl.requests, "post", fake_post)
    monkeypatch.setattr(orl.time, "sleep", lambda s: None)
    return calls, queue


def test_requires_key():
    with pytest.raises(ValueError):
        OpenRouterLLM("")


def test_success_path(posts):
    calls, queue = posts
    queue.append(_ok('{"answer": 1}', model="provider/real-model"))
    llm = OpenRouterLLM("sk-or-x", "vendor/m")
    out = llm.complete(model="ignored", system="s", user="u")
    assert out["text"] == '{"answer": 1}' and out["model"] == "provider/real-model"
    assert out["usage"]["input_tokens"] == 10 and out["usage"]["finish_reason"] == "stop"
    p = calls[0]["json"]
    assert p["model"] == "vendor/m" and p["response_format"] == {"type": "json_object"}
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-or-x"
    assert llm.totals["calls"] == 1


def test_json_mode_fallback_on_400(posts):
    calls, queue = posts
    queue += [Resp(400, {"error": {"message": "response_format not supported"}}), _ok("x")]
    out = OpenRouterLLM("k").complete(model="m", system="s", user="u")
    assert out["text"] == "x"
    assert "response_format" in calls[0]["json"] and "response_format" not in calls[1]["json"]


def test_non_transient_4xx_not_retried(posts):
    calls, queue = posts
    queue.append(Resp(401, {"error": {"message": "Invalid API key", "code": 401}}))
    with pytest.raises(OpenRouterError) as ei:
        OpenRouterLLM("bad").complete(model="m", system="s", user="u")
    assert ei.value.status == 401 and "Invalid API key" in str(ei.value)
    assert len(calls) == 1


def test_transient_retried_then_succeeds(posts):
    calls, queue = posts
    queue += [Resp(429, {"error": {"message": "slow down"}}), Resp(503, text="busy"), _ok("y")]
    out = OpenRouterLLM("k").complete(model="m", system="s", user="u")
    assert out["text"] == "y" and len(calls) == 3


def test_transient_gives_up_after_three(posts):
    calls, queue = posts
    queue += [Resp(502, text="bad gateway")] * 3
    with pytest.raises(OpenRouterError) as ei:
        OpenRouterLLM("k").complete(model="m", system="s", user="u")
    assert "after 3 attempts" in str(ei.value) and len(calls) == 3


def test_error_body_with_200(posts):
    calls, queue = posts
    queue.append(Resp(200, {"error": {"message": "No endpoints found", "code": 404}}))
    with pytest.raises(OpenRouterError) as ei:
        OpenRouterLLM("k").complete(model="m", system="s", user="u")
    assert ei.value.status == 404 and len(calls) == 1


def test_content_parts_are_joined(posts):
    _, queue = posts
    queue.append(Resp(200, {"choices": [{"message": {"content": [
        {"type": "text", "text": "{\"a\":"}, {"type": "text", "text": " 1}"}]}}]}))
    out = OpenRouterLLM("k").complete(model="m", system="s", user="u")
    assert out["text"] == '{"a": 1}' and out["usage"]["input_tokens"] == 0
