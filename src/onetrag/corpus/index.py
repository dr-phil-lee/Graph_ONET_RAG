"""Dense (sentence-transformers) + BM25 retrieval over the card corpus."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class CorpusIndex:
    def __init__(self, docs: list[dict], emb: np.ndarray, embedder_name: str,
                 query_prefix: str = ""):
        assert len(docs) == emb.shape[0]
        self.docs = docs
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.embedder_name = embedder_name
        self.query_prefix = query_prefix
        self._model = None
        self._bm25 = None

    # ------------------------------------------------------------ build/load

    @classmethod
    def build(cls, docs: list[dict], embedder_name: str, query_prefix: str = "",
              batch_size: int = 64, show_progress: bool = True) -> "CorpusIndex":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedder_name)
        emb = model.encode([d["text"] for d in docs], batch_size=batch_size,
                           show_progress_bar=show_progress, convert_to_numpy=True)
        idx = cls(docs, emb.astype(np.float32), embedder_name, query_prefix)
        idx._model = model
        return idx

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "emb.npy", self.emb)
        with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"embedder": self.embedder_name, "n_docs": len(self.docs),
                       "query_prefix": self.query_prefix}, f, indent=2)

    @classmethod
    def load(cls, out_dir: str | Path) -> "CorpusIndex":
        out = Path(out_dir)
        with open(out / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(out / "docs.jsonl", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        emb = np.load(out / "emb.npy")
        return cls(docs, emb, meta["embedder"], meta.get("query_prefix", ""))

    # --------------------------------------------------------------- search

    def _embed_query(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedder_name)
        v = self._model.encode([self.query_prefix + query], convert_to_numpy=True)[0]
        return v / np.linalg.norm(v)

    def search(self, query: str, k: int = 24) -> list[tuple[dict, float]]:
        v = self._embed_query(query)
        scores = self.emb @ v
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]

    def search_bm25(self, query: str, k: int = 24) -> list[tuple[dict, float]]:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in self.docs])
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]

    def search_rrf(self, query: str, k: int = 24, c: int = 60) -> list[tuple[dict, float]]:
        """Reciprocal-rank fusion of dense + BM25 (the strong S1+ baseline)."""
        fused: dict[str, float] = {}
        by_id = {}
        for results in (self.search(query, k * 2), self.search_bm25(query, k * 2)):
            for rank, (doc, _) in enumerate(results):
                fused[doc["doc_id"]] = fused.get(doc["doc_id"], 0.0) + 1.0 / (c + rank + 1)
                by_id[doc["doc_id"]] = doc
        top = sorted(fused.items(), key=lambda t: -t[1])[:k]
        return [(by_id[i], s) for i, s in top]
