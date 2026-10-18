"""REST API tests (FastAPI TestClient, fake LLM, no network)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api as api_mod
import bootstrap
from openrouter_llm import OpenRouterError
from tests.conftest import FakeLLM

RN = "29-1141.00"


@pytest.fixture(scope="module")
def client(engine):
    api_mod.app.state.engine = engine          # skip the 15 s load in lifespan
    with TestClient(api_mod.app) as c:
        yield c


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
    assert client.get("/openapi.json").status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["nodes"] == 30328 + 776 and body["edges"] == 236125 + 4171
    assert body["occupations"] == 894 + 29 and body["onet_release"] == "30.3"


def test_occupation_search_and_profile(client):
    r = client.get("/occupations", params={"q": "registered nurse", "limit": 3})
    assert r.status_code == 200 and r.json()[0]["soc"] == RN
    r = client.get(f"/occupations/{RN}")
    assert r.status_code == 200 and r.json()["name"] == "Registered Nurses"
    assert client.get("/occupations/00-0000.00").status_code == 404
    assert client.get("/occupations", params={"q": ""}).status_code == 422


def test_entities_dry_run(client):
    r = client.get("/entities", params={"question": "Compare nurses and pharmacists"})
    assert r.status_code == 200
    body = r.json()
    ids = {e["node_id"] for e in body["entities"]}
    assert RN in ids and "29-1051.00" in ids
    assert body["caption"].startswith("Graph entities:")
    r = client.get("/entities", params={"question": "x", "occupations": [RN]})
    assert r.json()["entities"][0]["source"] == "pinned"


def test_ask_grounded(client, fake_llm):
    r = client.post("/ask", json={"question": "What is the Job Zone of Registered Nurses?",
                                  "include_graph": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "grounded" and body["grounded"] is True
    assert body["seeds"] == [RN] and body["citations"]
    assert body["evidence"][0]["id"] == f"N-{RN}"
    assert body["graph"]["edges"] and body["graph"]["nodes"]
    assert "raw" not in body            # exclude_none: only present when unparsed
    assert body["elapsed_s"] >= 0


def test_ask_without_evidence_omits_it(client, fake_llm):
    r = client.post("/ask", json={"question": "tasks of Registered Nurses",
                                  "include_evidence": False})
    assert r.status_code == 200 and "evidence" not in r.json()


def test_ask_validation(client, fake_llm):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "budget": 10}).status_code == 422
    r = client.post("/ask", json={"question": "x", "occupations": ["99-9999.99"]})
    assert r.status_code == 422 and "99-9999.99" in r.json()["detail"]


def test_ask_pinned_occupation(client, fake_llm):
    r = client.post("/ask", json={"question": "most important abilities?",
                                  "occupations": [RN], "auto_link": False})
    assert r.status_code == 200
    # term is None for pins and omitted by response_model_exclude_none
    assert r.json()["entities"] == [{"node_id": RN, "name": "Registered Nurses",
                                     "label": "Job", "source": "pinned",
                                     "confidence": "high"}]


def test_ask_header_key_overrides_server_key(client, engine, monkeypatch):
    import engine as engine_mod

    captured = {}

    class Spy(FakeLLM):
        def __init__(self, api_key, model=None, **kw):
            super().__init__()
            captured["key"], captured["model"] = api_key, model
            self.model = model

    monkeypatch.setattr(engine_mod, "OpenRouterLLM", Spy)
    r = client.post("/ask", json={"question": "Job Zone of Registered Nurses?",
                                  "model": "vendor/model"},
                    headers={"X-OpenRouter-Key": "sk-or-header"})
    assert r.status_code == 200 and r.json()["model"] == "vendor/model"
    assert captured == {"key": "sk-or-header", "model": "vendor/model"}


def test_ask_without_any_key_is_401(client, engine):
    from engine import _MissingKeyLLM

    prev = engine.default_llm
    engine.default_llm = _MissingKeyLLM()
    try:
        r = client.post("/ask", json={"question": "Job Zone of Registered Nurses?"})
    finally:
        engine.default_llm = prev
    assert r.status_code == 401 and "OPENROUTER_API_KEY" in r.json()["detail"]


def test_openrouter_errors_are_mapped(client, engine):
    class Boom(FakeLLM):
        def __init__(self, status):
            super().__init__()
            self.status = status

        def complete(self, **kw):
            raise OpenRouterError("HTTP x: provider says no", self.status)

    prev = engine.default_llm
    try:
        for status, expected in ((401, 401), (402, 402), (404, 404), (500, 502), (None, 502)):
            engine.default_llm = Boom(status)
            r = client.post("/ask", json={"question": "Job Zone of Registered Nurses?"})
            assert r.status_code == expected, (status, r.text)
            assert "provider says no" in r.json()["detail"]
    finally:
        engine.default_llm = prev


def test_graph_endpoints(client):
    assert client.get("/graph/stats").json()["edges_by_type"]["RATED"] == 143914
    node = client.get(f"/graph/nodes/{RN}").json()
    assert node["label"] == "Job"
    assert client.get("/graph/nodes/nope").status_code == 404
    ego = client.get("/graph/ego", params={"node": RN, "edge_types": ["HAS_TASK"],
                                           "max_edges": 3}).json()
    assert len(ego["edges"]) == 3 and ego["center"] == RN
    eid = ego["edges"][0]["id"]
    edge = client.get(f"/graph/edges/{eid}").json()
    assert edge["type"] == "HAS_TASK" and edge["src"] == RN
    assert client.get("/graph/edges/-5").status_code == 404
    assert client.get("/graph/ego", params={"node": "nope"}).status_code == 404


def test_models_endpoint_degrades_offline(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(api_mod, "list_models", boom)
    api_mod.app.state.models_cache = (0.0, [])
    body = client.get("/models").json()
    assert body["models"] == [] and "error" in body
    assert body["default"] == bootstrap.OPENROUTER_DEFAULT_MODEL


def test_bearer_token_enforced(client, monkeypatch):
    monkeypatch.setattr(bootstrap, "API_TOKEN", "secret")
    assert client.get("/health").status_code == 200          # health is public
    assert client.get("/graph/stats").status_code == 401
    assert client.get("/graph/stats", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/graph/stats", headers={"Authorization": "Bearer secret"}).status_code == 200
