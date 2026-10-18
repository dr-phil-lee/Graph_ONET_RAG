"""REST API for the O*NET Graph RAG chatbot (FastAPI).

Run:   uvicorn api:app --host 0.0.0.0 --port 8000      (from the Chatbot folder)
Docs:  http://localhost:8000/docs   (interactive OpenAPI UI)

Every endpoint is a thin wrapper over engine.ChatEngine, which is loaded
once at startup and shared (read-only) by all requests. Per-request
settings — OpenRouter key, model, evidence budget, pinned occupations —
never leak between callers.

Authentication
- Server access: if CHATBOT_API_TOKEN is set in .env, every request must send
  `Authorization: Bearer <token>`.
- Generator key: the OpenRouter key is taken from (in order) the
  `X-OpenRouter-Key` header, the `api_key` field of the request body, or the
  server's OPENROUTER_API_KEY. The key is used only for that one request.

Conversation memory
- POST /ask with "session_id" (any id you choose, or one returned by an
  earlier call; "new" asks the server to create one). The server remembers
  each turn and the next question is read with the previous turns, so
  follow-ups like "which of those is least exposed to AI?" work.
- Stateless alternative: send "history" (the "memory_turn" objects returned
  by earlier /ask calls) yourself.
- GET /sessions/{id} shows what is remembered; DELETE /sessions/{id} forgets.

AI exposure
- Answers carry "ai_exposure" rows and occupation cards with the
  Opportunity Data Occupation AI Exposure Index (third-party, not O*NET),
  reported as scores only (0 = protected, 1 = exposed), each with its own
  calculation. "ai_exposure_method" (first field after the question)
  explains how the scores were computed and lists the references.
  GET /ai-exposure and GET /ai-exposure/{soc} expose the index itself.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import ai_exposure
import memory
from engine import ChatEngine, NotConfiguredError
from openrouter_llm import OpenRouterError, list_models

API_VERSION = "1.1.0"


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Natural-language question (English works best).")
    occupations: list[str] = Field(
        default_factory=list,
        description="O*NET-SOC codes to pin as retrieval seeds, e.g. ['29-1141.00'].")
    auto_link: bool = Field(
        True, description="Also fuzzy-match occupations from the question wording.")
    model: str | None = Field(None, description="OpenRouter model id; default from server.")
    api_key: str | None = Field(
        None, description="OpenRouter key for this request (prefer the X-OpenRouter-Key header).")
    budget: int | None = Field(
        None, ge=bootstrap.MIN_CONTEXT_BUDGET, le=bootstrap.MAX_CONTEXT_BUDGET,
        description="Evidence budget in tokens placed in front of the model.")
    include_evidence: bool = Field(True, description="Return every evidence line shown to the model.")
    include_graph: bool = Field(False, description="Return the evidence subgraph (nodes + edges).")
    include_focus_graph: bool = Field(
        True, description="Return the answer-focused subgraph (cited records + links "
                          "to recommended occupations).")
    session_id: str | None = Field(
        None, max_length=64,
        description="Conversation id for server-side memory ('new' = create one). "
                    "Letters, digits, '-' and '_' only.")
    history: list[dict] | None = Field(
        None, description="Stateless memory: earlier 'memory_turn' objects, oldest first. "
                          "Ignored when session_id is given.")
    use_memory: bool = Field(True, description="Read the question with the conversation so far.")


class EntityOut(BaseModel):
    node_id: str
    name: str
    label: str
    source: str = Field(description="pinned | linked (exact) | matched (fuzzy) | inferred (LLM) "
                                    "| memory (carried over from the conversation)")
    term: str | None = None
    confidence: str = "high"


class EvidenceOut(BaseModel):
    id: str
    statement: str
    cited: bool


class AskResponse(BaseModel):
    question: str
    standalone_question: str | None = Field(
        None, description="The follow-up rewritten with the conversation (memory).")
    ai_exposure_method: dict | None = Field(
        None, description="How the AI exposure scores were computed: summary, formula, "
                          "steps, the nine O*NET variables, limitations, join rule and "
                          "references. Present whenever an occupation in focus has a score.")
    headline: str = Field("", description="One-sentence direct answer.")
    key_points: list[str] = Field(default_factory=list, description="Up to 5 cited findings.")
    answer: Any = None
    occupations: list[dict] = Field(
        default_factory=list,
        description="Occupation cards (recommended / compared): soc, title, job_zone, "
                    "description, related_rank, related_tier, top_skills, ai_exposure, "
                    "fit (model text), citations. Only occupations present in the evidence.")
    ai_exposure: list[dict] = Field(
        default_factory=list,
        description="AI Exposure Index rows for the occupations in focus: soc, title, "
                    "role (current | recommended | discussed), exposure (null = not covered; "
                    "composite, digital, human_contact, physical, calculation, ...).")
    follow_ups: list[str] = Field(default_factory=list, description="Suggested next questions.")
    reasoning: str | None = None
    citations: list[str]
    invalid_citations: list[str]
    status: str = Field(description="grounded | partially_grounded | ungrounded | truncated | unparsed")
    grounded: bool
    truncated: bool
    model: str | None
    entities: list[EntityOut]
    sources: list[dict] = Field(
        default_factory=list,
        description="One record per valid citation: id, statement, O*NET 30.3 table, nodes.")
    seeds: list[str]
    cited_edge_ids: list[int]
    retrieved_edge_ids: list[int]
    focus_edge_ids: list[int] = Field(default_factory=list)
    focus_node_ids: list[str] = Field(default_factory=list)
    focus_graph: dict | None = Field(None, description="Answer-focused subgraph (nodes + edges).")
    history_turns: int = Field(0, description="Prior turns the answer could see.")
    session_id: str | None = None
    session_turns: int | None = Field(None, description="Turns stored in the session after this one.")
    memory_turn: dict | None = Field(
        None, description="This turn as a memory record; send it back in 'history' "
                          "for stateless multi-turn use.")
    context_tokens: int
    usage: dict
    raw: str | None = Field(None, description="Raw model text, only when it was not valid JSON.")
    evidence: list[EvidenceOut] | None = None
    graph: dict | None = None
    elapsed_s: float


class EntitiesResponse(BaseModel):
    question: str
    entities: list[EntityOut]
    seeds: list[str]
    suggestions: dict[str, list[dict]]
    caption: str


class OccupationHit(BaseModel):
    soc: str
    title: str
    matched_on: str
    score: float


class HealthResponse(BaseModel):
    status: str
    version: str
    onet_release: str | None
    nodes: int
    edges: int
    occupations: int
    default_model: str
    server_key_configured: bool
    auth_required: bool
    default_budget: int


# ----------------------------------------------------------------- startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "engine", None) is None:  # tests may pre-inject one
        app.state.engine = ChatEngine()
    app.state.models_cache = (0.0, [])
    if getattr(app.state, "sessions", None) is None:
        app.state.sessions = memory.ConversationStore()
    yield


app = FastAPI(
    title="O*NET Graph RAG API",
    version=API_VERSION,
    description=(
        "Graph-traversal RAG over the O*NET 30.3 occupational knowledge graph. "
        "Every answer cites the individual database records (graph edges) it "
        "was derived from, and every citation is validated server-side."),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=bootstrap.API_CORS_ORIGINS,
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def get_engine(request: Request) -> ChatEngine:
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(503, "engine not loaded yet")
    return eng


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not bootstrap.API_TOKEN:
        return
    expected = f"Bearer {bootstrap.API_TOKEN}"
    if authorization != expected:
        raise HTTPException(401, "missing or invalid bearer token")


Auth = Depends(require_token)


# --------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(engine: ChatEngine = Depends(get_engine)):
    meta = engine.graph_stats()
    return HealthResponse(
        status="ok", version=API_VERSION, onet_release=meta.get("onet_release"),
        nodes=engine.G.number_of_nodes(), edges=engine.G.number_of_edges(),
        occupations=len(engine.occupations), default_model=engine.default_model,
        server_key_configured=engine.is_configured,
        auth_required=bool(bootstrap.API_TOKEN), default_budget=engine.budget)


@app.get("/models", tags=["meta"], dependencies=[Auth])
def models(request: Request):
    """OpenRouter model ids (live list, cached for an hour; empty on network error)."""
    ts, cached = request.app.state.models_cache
    if time.time() - ts < 3600 and cached:
        return {"default": bootstrap.OPENROUTER_DEFAULT_MODEL, "models": cached}
    try:
        ids = list_models()
    except Exception as e:  # noqa: BLE001 — offline is not an API failure
        return {"default": bootstrap.OPENROUTER_DEFAULT_MODEL, "models": [],
                "error": f"could not fetch the OpenRouter catalog: {e}"}
    request.app.state.models_cache = (time.time(), ids)
    return {"default": bootstrap.OPENROUTER_DEFAULT_MODEL, "models": ids}


@app.get("/occupations", response_model=list[OccupationHit], tags=["lookup"],
         dependencies=[Auth])
def occupations(q: str = Query(..., min_length=1, max_length=200,
                               description="Title, alternate title or fragment"),
                limit: int = Query(10, ge=1, le=100),
                engine: ChatEngine = Depends(get_engine)):
    """Find O*NET-SOC codes by title or alternate title (exact > substring > fuzzy)."""
    return engine.search_occupations(q, limit)


@app.get("/occupations/{soc}", tags=["lookup"], dependencies=[Auth])
def occupation(soc: str, engine: ChatEngine = Depends(get_engine)):
    """Full profile of one occupation node (description, Job Zone, alternate titles, edge counts)."""
    prof = engine.node_profile(soc)
    if prof is None or prof["label"] != "Job":
        raise HTTPException(404, f"unknown O*NET-SOC code: {soc}")
    prof["ai_exposure"] = engine.exposure_for(soc)
    return prof


@app.get("/entities", response_model=EntitiesResponse, tags=["lookup"],
         dependencies=[Auth])
def entities(question: str = Query(..., min_length=1, max_length=2000),
             occupations: list[str] = Query(default=[], description="SOC codes to pin"),
             auto_link: bool = True,
             engine: ChatEngine = Depends(get_engine)):
    """Dry run of entity linking: which graph nodes would seed retrieval for this question."""
    res = engine.resolve_seeds(question, pinned=occupations, auto_match=auto_link)
    return EntitiesResponse(
        question=question,
        entities=[EntityOut(**e.__dict__) for e in res.entities],
        seeds=res.seeds,
        suggestions={term: [{"soc": s, "title": t} for s, t in cands]
                     for term, cands in res.suggestions.items()},
        caption=res.caption())


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True,
          tags=["chat"], dependencies=[Auth])
def ask(req: AskRequest, request: Request,
        x_openrouter_key: str | None = Header(default=None, alias="X-OpenRouter-Key"),
        engine: ChatEngine = Depends(get_engine)):
    """Answer a question with graph evidence and validated, record-level citations."""
    bad = [s for s in req.occupations if s not in engine.G
           or engine.G.nodes[s].get("label") != "Job"]
    if bad:
        raise HTTPException(422, f"unknown O*NET-SOC code(s): {bad}")
    store: memory.ConversationStore = request.app.state.sessions
    sid = req.session_id
    if sid == "new":
        sid = store.new_id()
    elif sid is not None and not store.valid_id(sid):
        raise HTTPException(422, "session_id may contain only letters, digits, '-' and '_' "
                                 "(max 64)")
    if not req.use_memory:
        history = []
    elif sid is not None:
        history = store.get(sid)
    else:
        history = memory.normalize_turns(req.history)
    t0 = time.perf_counter()
    try:
        ans = engine.ask(req.question, forced_seeds=req.occupations,
                         api_key=x_openrouter_key or req.api_key, model=req.model,
                         budget=req.budget, auto_link=req.auto_link, history=history)
    except NotConfiguredError as e:
        raise HTTPException(401, str(e)) from e
    except OpenRouterError as e:
        status = e.status if e.status in (401, 402, 403, 404, 429) else 502
        raise HTTPException(status, f"OpenRouter: {e}") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    body = ans.to_dict(include_evidence=req.include_evidence)
    if req.include_graph:
        body["graph"] = engine.subgraph(ans.retrieved_edge_ids, set(ans.cited_edge_ids))
    if req.include_focus_graph:
        fg = engine.subgraph(ans.focus_edge_ids, set(ans.cited_edge_ids))
        have = {n["id"] for n in fg["nodes"]}
        for n in ans.focus_node_ids:
            if n not in have:
                name, label = engine.node_info(n)
                fg["nodes"].append({"id": n, "name": name, "label": label})
        for n in fg["nodes"]:
            if n["label"] == "Job":
                n["ai_exposure"] = engine.exposure_for(n["id"])
        body["focus_graph"] = fg
    turn = ans.memory_turn()
    body["memory_turn"] = turn
    if sid is not None:
        body["session_id"] = sid
        body["session_turns"] = store.append(sid, turn)
    body["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return body


@app.get("/sessions/{session_id}", tags=["chat"], dependencies=[Auth])
def get_session(session_id: str, request: Request):
    """What the server remembers for one conversation."""
    store: memory.ConversationStore = request.app.state.sessions
    if not store.exists(session_id):
        raise HTTPException(404, f"unknown session: {session_id}")
    turns = store.get(session_id)
    return {"session_id": session_id, "turns": turns, "count": len(turns)}


@app.delete("/sessions/{session_id}", tags=["chat"], dependencies=[Auth])
def delete_session(session_id: str, request: Request):
    """Forget a conversation."""
    store: memory.ConversationStore = request.app.state.sessions
    if not store.delete(session_id):
        raise HTTPException(404, f"unknown session: {session_id}")
    return {"session_id": session_id, "deleted": True}


@app.get("/ai-exposure", tags=["ai exposure"], dependencies=[Auth])
def ai_exposure_about():
    """Source, methodology, references and score distribution of the AI Exposure Index."""
    return ai_exposure.about()


@app.get("/ai-exposure/{soc}", tags=["ai exposure"], dependencies=[Auth])
def ai_exposure_one(soc: str, engine: ChatEngine = Depends(get_engine)):
    """AI exposure of one O*NET occupation (404 when the index does not cover it)."""
    if soc not in engine.G or engine.G.nodes[soc].get("label") != "Job":
        raise HTTPException(404, f"unknown O*NET-SOC code: {soc}")
    x = engine.exposure_for(soc)
    if x is None:
        raise HTTPException(404, f"{soc} is not covered by the AI Exposure Index")
    return {"soc": soc, "title": engine.G.nodes[soc].get("name"), **x}


@app.get("/graph/stats", tags=["graph"], dependencies=[Auth])
def graph_stats(engine: ChatEngine = Depends(get_engine)):
    """Node and edge counts by type (from data/graph/meta.json)."""
    return engine.graph_stats()


@app.get("/graph/nodes/{node_id}", tags=["graph"], dependencies=[Auth])
def graph_node(node_id: str, engine: ChatEngine = Depends(get_engine)):
    prof = engine.node_profile(node_id)
    if prof is None:
        raise HTTPException(404, f"unknown node id: {node_id}")
    return prof


@app.get("/graph/edges/{edge_id}", tags=["graph"], dependencies=[Auth])
def graph_edge(edge_id: int, engine: ChatEngine = Depends(get_engine)):
    """One graph edge = one citable O*NET record (the 'E123' ids in answers)."""
    rec = engine.edge_record(edge_id)
    if rec is None:
        raise HTTPException(404, f"unknown edge id: {edge_id}")
    return rec


@app.get("/graph/ego", tags=["graph"], dependencies=[Auth])
def graph_ego(node: str = Query(..., description="center node id, e.g. 29-1141.00"),
              radius: int = Query(1, ge=1, le=3),
              edge_types: list[str] = Query(default=[], description="e.g. HAS_TASK, RATED"),
              max_edges: int = Query(150, ge=1, le=1000),
              engine: ChatEngine = Depends(get_engine)):
    """Ego network around any node, as JSON."""
    try:
        return engine.ego_subgraph(node, radius=radius,
                                   edge_types=set(edge_types) or None, max_edges=max_edges)
    except KeyError:
        raise HTTPException(404, f"unknown node id: {node}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
