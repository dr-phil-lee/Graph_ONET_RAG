"""Shared fixtures: one ChatEngine per test session (graph load ~15 s) and a
deterministic fake LLM so the whole stack runs without an OpenRouter key."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bootstrap  # noqa: E402,F401
from engine import ChatEngine  # noqa: E402


class FakeLLM:
    """Cites the first `n_cite` evidence ids it is shown, plus whatever extra
    tokens the test injects, so citation validation can be exercised."""

    description = "fake"
    model = "fake/model"

    def __init__(self, n_cite: int = 3, extra_citations: list | None = None,
                 answer="ok", raw_override: str | None = None,
                 link_reply: list[str] | None = None,
                 context_reply: dict | None = None):
        self.n_cite = n_cite
        self.extra = extra_citations or []
        self.answer = answer
        self.raw_override = raw_override
        self.link_reply = link_reply          # occupations returned to the link prompt
        self.link_calls = 0
        self.context_reply = context_reply   # reply to the memory-rewrite prompt
        self.context_calls = 0
        self.last_context_user: str | None = None
        self.last_user: str | None = None
        self.last_system: str | None = None

    def complete(self, *, model, system, user, **kw) -> dict:
        if system.startswith("Extract the occupations"):
            self.link_calls += 1
            return {"text": json.dumps({"occupations": self.link_reply or []}),
                    "usage": {}, "model": self.model}
        if system.startswith("You rewrite"):
            self.context_calls += 1
            self.last_context_user = user
            return {"text": json.dumps(self.context_reply or {}), "usage": {},
                    "model": self.model}
        self.last_user, self.last_system = user, system
        if self.raw_override is not None:
            text = self.raw_override
        else:
            ids = re.findall(r"^\[([^\]]+)\]", user, flags=re.M)[: self.n_cite]
            text = json.dumps({"answer": self.answer, "citations": ids + self.extra,
                               "reasoning": "fake"})
        return {"text": text, "model": self.model,
                "usage": {"input_tokens": len(user) // 4, "output_tokens": len(text) // 4,
                          "latency_s": 0.01}, "cached": False}

    def count_tokens(self, text: str, model=None):
        return max(1, len(text) // 4), "estimate"


@pytest.fixture(scope="session")
def engine() -> ChatEngine:
    eng = ChatEngine()
    eng.default_llm = FakeLLM()
    eng.rag = eng._make_rag(eng.default_llm, eng.budget)
    return eng


@pytest.fixture
def fake_llm(engine):
    """Swap in a fresh FakeLLM for one test; restored afterwards."""
    prev = engine.default_llm
    llm = FakeLLM()
    engine.default_llm = llm
    yield llm
    engine.default_llm = prev
