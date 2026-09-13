"""S0 — zero-shot: no retrieval. Measures the generator's parametric O*NET
knowledge; the reported effect of every RAG system is its delta over S0."""
from __future__ import annotations

from .base import RAGSystem


class ZeroShot(RAGSystem):
    name = "s0_zeroshot"

    def retrieve(self, item: dict) -> list[dict]:
        return []
