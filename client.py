"""Minimal Python client for the O*NET Graph RAG REST API (api.py).

    from client import OnetGraphRagClient
    c = OnetGraphRagClient("http://localhost:8000", openrouter_key="sk-or-...")
    r = c.ask("What is the Job Zone of Registered Nurses?")
    print(r["headline"], r["answer"], r["citations"], r["status"])

    # multi-turn chat with server-side memory
    chat = c.conversation()
    r1 = chat.ask("I was a telemarketer and lost my job to AI. What else could I do?")
    r2 = chat.ask("Which of those is least exposed to AI?")   # remembers r1

Only `requests` is needed. Run this file directly for a demo:
    python client.py "Which tasks of Registered Nurses involve medication?"
"""
from __future__ import annotations

import sys

import requests


class OnetGraphRagClient:
    def __init__(self, base_url: str = "http://localhost:8000",
                 openrouter_key: str | None = None, api_token: str | None = None,
                 timeout_s: float = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._headers: dict[str, str] = {}
        if openrouter_key:
            self._headers["X-OpenRouter-Key"] = openrouter_key
        if api_token:
            self._headers["Authorization"] = f"Bearer {api_token}"

    # --------------------------------------------------------------- http
    def _get(self, path: str, **params):
        r = requests.get(f"{self.base_url}{path}", params=params,
                         headers=self._headers, timeout=self.timeout_s)
        self._raise(r)
        return r.json()

    def _post(self, path: str, body: dict):
        r = requests.post(f"{self.base_url}{path}", json=body,
                          headers=self._headers, timeout=self.timeout_s)
        self._raise(r)
        return r.json()

    @staticmethod
    def _raise(r: requests.Response) -> None:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")

    # ------------------------------------------------------------ methods
    def health(self) -> dict:
        return self._get("/health")

    def models(self) -> dict:
        return self._get("/models")

    def search_occupations(self, text: str, limit: int = 10) -> list[dict]:
        return self._get("/occupations", q=text, limit=limit)

    def occupation(self, soc: str) -> dict:
        return self._get(f"/occupations/{soc}")

    def entities(self, question: str, occupations: list[str] | None = None,
                 auto_link: bool = True) -> dict:
        return self._get("/entities", question=question,
                         occupations=occupations or [], auto_link=auto_link)

    def ask(self, question: str, occupations: list[str] | None = None, *,
            model: str | None = None, budget: int | None = None,
            auto_link: bool = True, include_evidence: bool = True,
            include_graph: bool = False, include_focus_graph: bool = True,
            session_id: str | None = None, history: list[dict] | None = None,
            use_memory: bool = True) -> dict:
        """session_id: server-side memory ("new" creates one; the response
        carries the id). history: stateless memory (earlier "memory_turn"
        objects)."""
        body = {"question": question, "occupations": occupations or [],
                "auto_link": auto_link, "include_evidence": include_evidence,
                "include_graph": include_graph,
                "include_focus_graph": include_focus_graph, "use_memory": use_memory}
        if model:
            body["model"] = model
        if budget:
            body["budget"] = budget
        if session_id:
            body["session_id"] = session_id
        elif history:
            body["history"] = history
        return self._post("/ask", body)

    def conversation(self, session_id: str = "new", **defaults) -> "Conversation":
        """A chat whose turns the server remembers."""
        return Conversation(self, session_id, defaults)

    def session(self, session_id: str) -> dict:
        return self._get(f"/sessions/{session_id}")

    def forget(self, session_id: str) -> dict:
        r = requests.delete(f"{self.base_url}/sessions/{session_id}",
                            headers=self._headers, timeout=self.timeout_s)
        self._raise(r)
        return r.json()

    def ai_exposure(self, soc: str | None = None) -> dict:
        """Index metadata, or one occupation's AI exposure."""
        return self._get(f"/ai-exposure/{soc}" if soc else "/ai-exposure")

    def graph_stats(self) -> dict:
        return self._get("/graph/stats")

    def node(self, node_id: str) -> dict:
        return self._get(f"/graph/nodes/{node_id}")

    def edge(self, edge_id: int) -> dict:
        return self._get(f"/graph/edges/{edge_id}")

    def ego(self, node: str, radius: int = 1, edge_types: list[str] | None = None,
            max_edges: int = 150) -> dict:
        return self._get("/graph/ego", node=node, radius=radius,
                         edge_types=edge_types or [], max_edges=max_edges)


class Conversation:
    """Multi-turn helper: every ask() reuses the same server session."""

    def __init__(self, client: OnetGraphRagClient, session_id: str = "new",
                 defaults: dict | None = None):
        self.client = client
        self.session_id = session_id
        self.defaults = defaults or {}
        self.turns: list[dict] = []

    def ask(self, question: str, **kw) -> dict:
        res = self.client.ask(question, session_id=self.session_id,
                              **{**self.defaults, **kw})
        self.session_id = res.get("session_id", self.session_id)
        self.turns.append(res)
        return res

    def forget(self) -> dict:
        return self.client.forget(self.session_id)


if __name__ == "__main__":
    import json
    import os

    question = " ".join(sys.argv[1:]) or "What is the Job Zone of Registered Nurses?"
    c = OnetGraphRagClient(os.getenv("CHATBOT_API_URL", "http://localhost:8000"),
                           openrouter_key=os.getenv("OPENROUTER_API_KEY"),
                           api_token=os.getenv("CHATBOT_API_TOKEN"))
    print("health:", json.dumps(c.health(), indent=1))
    res = c.ask(question, include_evidence=False)
    print(json.dumps({k: res.get(k) for k in ("headline", "key_points", "answer",
                                              "occupations", "ai_exposure", "follow_ups",
                                              "reasoning", "citations", "invalid_citations",
                                              "status", "model", "entities",
                                              "context_tokens", "elapsed_s")},
                     indent=1, ensure_ascii=False))
