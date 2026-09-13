"""Shared RAG-system interface: retrieve context units, generate a JSON answer.

Fairness contract enforced here for every system:
- identical generator model, temperature, and answer-format instructions;
- context filled greedily in each system's own rank order and truncated at
  unit boundaries to the shared token budget B (cfg.budget.context_tokens);
- realized context tokens recorded per question for reporting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..llm.client import LLM

SYSTEM_PROMPT = """You are an expert on the O*NET occupational database (release 30.3).
Answer the question using the evidence provided (if any). Respond with ONLY a JSON object:
{"answer": <string, number, boolean, or array of strings>,
 "citations": <array of evidence ids you relied on, e.g. ["E123", "D-task-..."]; [] if no evidence given>,
 "reasoning": <one or two sentences>}
Rules:
- For set questions, return an array of names. For numeric questions, return the number.
- For yes/no questions, return true or false.
- Cite only evidence ids that appear in the provided context.
- If the evidence does not contain the answer, still give your best answer but cite nothing."""


class RAGSystem(ABC):
    name: str = "base"

    def __init__(self, cfg: dict, llm: LLM):
        self.cfg = cfg
        self.llm = llm
        self.budget = int(cfg["budget"]["context_tokens"])

    @abstractmethod
    def retrieve(self, item: dict) -> list[dict]:
        """Ranked context units: {"unit_id": str, "text": str, "edge_ids": [int]}."""

    def answer(self, item: dict, model: str | None = None) -> dict:
        units = self.retrieve(item)
        # greedy fill with a local chars/4 estimate (per-unit API counting would
        # cost one request per unit); the realized context is API-measured once
        kept, used = [], 0
        for u in units:
            n = max(1, len(u["text"]) // 4)
            if used + n > self.budget and kept:
                break
            kept.append(u)
            used += n
        method = "estimate"
        if kept:
            try:
                used, method = self.llm.count_tokens("\n".join(u["text"] for u in kept))
            except Exception:  # noqa: BLE001 — reporting only; keep the estimate
                pass
        question = item.get("question_natural") or item["question_template"]
        ctx = ""
        if kept:
            ctx = "\n".join(f"[{u['unit_id']}] {u['text']}" if not u["text"].startswith("[")
                            else u["text"] for u in kept)
            user = f"Evidence:\n{ctx}\n\nQuestion: {question}"
        else:
            user = f"Question: {question}"
        resp = self.llm.complete(model=model or self.cfg["models"]["generator"],
                                 system=SYSTEM_PROMPT, user=user)
        return {
            "qid": item["qid"], "system": self.name, "raw": resp["text"],
            "usage": resp["usage"], "cached": resp.get("cached", False),
            "context_units": [u["unit_id"] for u in kept],
            "unit_edge_ids": {u["unit_id"]: u.get("edge_ids", []) for u in kept},
            "unit_node_ids": {u["unit_id"]: u.get("node_ids", []) for u in kept},
            "context_edge_ids": sorted({e for u in kept for e in u.get("edge_ids", [])}),
            "context_text": ctx, "context_tokens": used, "token_method": method,
        }
