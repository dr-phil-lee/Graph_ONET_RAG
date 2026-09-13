"""S1 — vector RAG over the card corpus (dense; S1+ adds BM25 fusion)."""
from __future__ import annotations

from ..corpus.index import CorpusIndex
from ..llm.client import LLM
from .base import RAGSystem


class VectorRAG(RAGSystem):
    name = "s1_vector"

    def __init__(self, cfg: dict, llm: LLM, index: CorpusIndex, mode: str = "dense"):
        super().__init__(cfg, llm)
        self.index = index
        self.mode = mode
        if mode == "rrf":
            self.name = "s1p_vector_rrf"

    def retrieve(self, item: dict) -> list[dict]:
        question = item.get("question_natural") or item["question_template"]
        k = int(self.cfg["retrieval"]["vector_top_k"])
        search = self.index.search_rrf if self.mode == "rrf" else self.index.search
        return [{"unit_id": doc["doc_id"], "text": doc["text"],
                 "edge_ids": doc.get("edge_ids", []), "node_ids": doc.get("node_ids", [])}
                for doc, _ in search(question, k)]
