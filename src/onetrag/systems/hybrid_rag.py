"""S3 — hybrid: vector seed cards, then 1-hop graph enrichment around the
entities those cards mention."""
from __future__ import annotations

import networkx as nx

from ..corpus.index import CorpusIndex
from ..graph import query as q
from ..graph.serialize import LevelAnchors, edge_line
from ..llm.client import LLM
from .base import RAGSystem


class HybridRAG(RAGSystem):
    name = "s3_hybrid"

    def __init__(self, cfg: dict, llm: LLM, index: CorpusIndex,
                 G: nx.MultiDiGraph, anchors: LevelAnchors):
        super().__init__(cfg, llm)
        self.index = index
        self.G = G
        self.anchors = anchors

    def retrieve(self, item: dict) -> list[dict]:
        question = item.get("question_natural") or item["question_template"]
        m = int(self.cfg["retrieval"]["hybrid_seed_cards"])
        cards = self.index.search(question, m)
        units = [{"unit_id": doc["doc_id"], "text": doc["text"],
                  "edge_ids": doc.get("edge_ids", []),
                  "node_ids": doc.get("node_ids", [])} for doc, _ in cards]
        card_nodes = [n for doc, _ in cards for n in doc.get("node_ids", []) if n in self.G]
        seen_edges = {e for u in units for e in u["edge_ids"]}
        exp = q.khop_edges(self.G, card_nodes[:12], depth=1,
                           max_edges=int(self.cfg["graph"]["expansion"]["max_edges"]))
        for u, v, k, d in exp:
            if k in seen_edges:
                continue
            units.append({"unit_id": f"E{k}",
                          "text": edge_line(self.G, (u, v, k, d), self.anchors),
                          "edge_ids": [int(k)], "node_ids": [u, v]})
        return units
