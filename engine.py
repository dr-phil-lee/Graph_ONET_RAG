"""ChatEngine: one-time loading of graph + linker, per-question ask().

Wires the frozen research S2 stack (imported, never modified) to an
OpenRouter-hosted generator model. Config overrides happen in memory only —
config.yaml is never touched.

Design notes (shared by the Streamlit UI and the REST API):
- The heavy state (graph, entity linker, anchors, indexes) is loaded once and
  is read-only afterwards, so one ChatEngine can serve concurrent requests.
- Everything a caller may vary per request — API key, model, evidence
  budget, pinned occupations — is passed to ask(); a lightweight
  HybridGraphRAG is built per call, so sessions never leak settings into
  each other.
- resolve_seeds() is the single place where a question is turned into graph
  seeds (pins -> exact entity linker -> fuzzy occupation matcher), so the UI
  caption, the API response and the retrieval all agree.
- Every citation the model returns is checked against the evidence it was
  shown; invalid ids are reported, never silently dropped, and the answer
  carries an explicit grounding status.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
from onetrag.config import data_path, load_config
from onetrag.graph import query as q
from onetrag.graph import store
from onetrag.graph.serialize import LevelAnchors, edge_line
from onetrag.linking import EntityLinker
from onetrag.onet_io import OnetData

import ai_exposure
import graph_ext
import memory
from hybrid_graph_rag import HISTORY_TURNS, HybridGraphRAG, ai_question
from openrouter_llm import OpenRouterLLM


class NotConfiguredError(RuntimeError):
    """No OpenRouter API key available for this request."""


class _MissingKeyLLM:
    """Placeholder until an OpenRouter API key is supplied."""

    description = "not configured — no OpenRouter API key"
    model = None

    def complete(self, **kwargs) -> dict:
        raise NotConfiguredError(
            "No OpenRouter API key is set. Pass one per request, or put "
            "OPENROUTER_API_KEY in Chatbot/.env.")

    def count_tokens(self, text: str, model: str | None = None) -> tuple[int, str]:
        return max(1, len(text) // 4), "estimate"


# ------------------------------------------------------------------ results
@dataclass
class Entity:
    node_id: str
    name: str
    label: str
    source: str            # "pinned" | "linked" | "matched" | "inferred" | "memory"
    term: str | None = None  # question words (matched) or model-extracted name (inferred)
    confidence: str = "high"  # "low" = single word matched only through an alternate title


@dataclass
class SeedResolution:
    entities: list[Entity]
    seeds: list[str]                                   # ids handed to retrieval
    suggestions: dict[str, list[tuple[str, str]]]      # term -> [(soc, title)]

    @property
    def matched(self) -> list[Entity]:
        return [e for e in self.entities if e.source in ("matched", "inferred")]

    def caption(self) -> str:
        return entity_caption(self.entities)


_MARKS = {"pinned": "📌 ", "linked": "", "matched": "≈ ", "inferred": "≈ ", "memory": "🧠 "}


def entity_caption(entities: list[Entity]) -> str:
    if not entities:
        return ""
    names = " · ".join(f"{_MARKS.get(e.source, '')}{e.name} ({e.label})" for e in entities)
    notes = []
    if any(e.source in ("matched", "inferred") for e in entities):
        notes.append("≈ auto-matched from your wording")
    if any(e.source == "memory" for e in entities):
        notes.append("🧠 carried over from the conversation")
    note = ("  — " + "; ".join(notes) + ". If wrong, name the occupation exactly."
            if notes else "")
    return "Graph entities: " + names + note


@dataclass
class Answer:
    question: str
    parsed: dict | None                       # {"answer","citations","reasoning"} or None
    raw: str
    seeds: list[str]                          # node ids that seeded retrieval
    evidence: list[tuple[str, str, bool]]     # (unit_id, text, cited)
    cited_edge_ids: list[int]
    retrieved_edge_ids: list[int]
    usage: dict
    context_tokens: int
    citations: list[str] = field(default_factory=list)          # valid, normalized
    invalid_citations: list[str] = field(default_factory=list)  # ids not in the evidence
    status: str = "unparsed"   # grounded | partially_grounded | ungrounded | truncated | unparsed
    model: str | None = None
    entities: list[Entity] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # one per valid citation
    # structured view (all validated against the evidence / the graph)
    headline: str = ""
    key_points: list[str] = field(default_factory=list)
    occupations: list[dict] = field(default_factory=list)   # occupation cards
    follow_ups: list[str] = field(default_factory=list)
    ai_exposure: list[dict] = field(default_factory=list)   # focus jobs' index rows
    focus_edge_ids: list[int] = field(default_factory=list)  # answer-focused graph
    focus_node_ids: list[str] = field(default_factory=list)
    # conversation memory
    standalone_question: str | None = None   # follow-up rewritten with the history
    history_turns: int = 0                    # prior turns the answer could see
    # the question is about AI (or the answer cited an AI exposure line): only
    # then does the UI explain the score and its methodology
    ai_relevant: bool = False
    rec: dict = field(repr=False, default_factory=dict)

    @property
    def grounded(self) -> bool:
        return self.status == "grounded"

    @property
    def truncated(self) -> bool:
        return bool(self.parsed and self.parsed.get("_truncated"))

    @property
    def entity_caption(self) -> str:
        return entity_caption(self.entities)

    def memory_turn(self) -> dict:
        return memory.turn_from_answer(self)

    @property
    def exposure_in_evidence(self) -> bool:
        """AI exposure lines were retrieved for this question (AI or
        career-change questions only)."""
        return any(uid.startswith("X-") for uid, _, _ in self.evidence)

    @property
    def exposure_method(self) -> dict | None:
        """What the AI exposure score means and how it was computed (shown
        after the answer); None when no occupation in focus has a score."""
        if not any(r.get("exposure") for r in self.ai_exposure):
            return None
        return ai_exposure.methodology()

    @property
    def unsupported_occupations(self) -> list[str]:
        """Titles of listed occupations that are not backed by the retrieved
        evidence at all: the card cites nothing, the occupation was not a
        retrieval seed (its records were never retrieved), and no cited
        evidence line mentions it. An occupation the question was about
        (confirmed or linked) is backed by its retrieved profile even when
        the model forgot to list citations on its card."""
        in_graph = set(self.seeds) | {e.node_id for e in self.entities}
        cited_text = " ".join(text for _, text, cited in self.evidence if cited).lower()
        out = []
        for c in self.occupations:
            soc, title = c.get("soc") or "", (c.get("title") or "").lower()
            if c.get("citations") or soc in in_graph:
                continue
            if (soc and soc in cited_text) or (title and title in cited_text):
                continue
            out.append(c["title"])
        return out

    def to_dict(self, include_evidence: bool = True) -> dict:
        p = self.parsed or {}
        d = {
            "question": self.question,
            "standalone_question": self.standalone_question,
            "headline": self.headline,
            "key_points": list(self.key_points),
            "answer": p.get("answer"),
            "occupations": [dict(c) for c in self.occupations],
            "ai_exposure": [dict(x) for x in self.ai_exposure],
            "follow_ups": list(self.follow_ups),
            "reasoning": p.get("reasoning"),
            "citations": list(self.citations),
            "invalid_citations": list(self.invalid_citations),
            "status": self.status,
            "grounded": self.grounded,
            "truncated": self.truncated,
            "model": self.model,
            "entities": [asdict(e) for e in self.entities],
            "sources": [dict(s) for s in self.sources],
            "seeds": list(self.seeds),
            "cited_edge_ids": list(self.cited_edge_ids),
            "retrieved_edge_ids": list(self.retrieved_edge_ids),
            "focus_edge_ids": list(self.focus_edge_ids),
            "focus_node_ids": list(self.focus_node_ids),
            "history_turns": self.history_turns,
            "context_tokens": self.context_tokens,
            "usage": dict(self.usage),
            "raw": self.raw if self.parsed is None else None,
            # last: only meaningful when the question was about AI
            "ai_relevant": self.ai_relevant,
            "ai_exposure_method": self.exposure_method if self.ai_relevant else None,
        }
        if include_evidence:
            d["evidence"] = [{"id": uid, "statement": _strip_tag(uid, text), "cited": cited}
                             for uid, text, cited in self.evidence]
        return d


def _strip_tag(uid: str, text: str) -> str:
    return text[len(uid) + 3:] if text.startswith(f"[{uid}] ") else text


# O*NET 30.3 database files behind each kind of graph record (data attribution)
_ONET_TABLE = {
    "HAS_TASK": "Task Statements + Task Ratings (task_statements.csv, task_ratings.csv)",
    "MAPS_TO": "Tasks to DWAs (tasks_to_dwas.csv)",
    "PART_OF": "Work-activity hierarchy (gwas_to_iwas_to_dwas.csv)",
    "ENABLES": "Descriptor-to-work-activity crosswalks (*_to_work_activities.csv)",
    "RELATED_TO": "Related Occupations (related_occupations.csv)",
    "USES_TECH": "Technology Skills (software_skills.csv)",
    "IN_GROUP": "Content Model Reference (content_model_reference.csv)",
}
_RATED_TABLE = {
    "EssentialSkill": "Essential Skills ratings (essential_skills.csv)",
    "TransferableSkill": "Transferable Skills ratings (transferable_skills.csv)",
    "Ability": "Abilities ratings (abilities.csv)",
    "Knowledge": "Knowledge ratings (knowledge.csv)",
    "WorkStyle": "Work Styles ratings (work_styles.csv)",
    "GWA": "Work Activities ratings (work_activities.csv)",
}
_PROFILE_TABLE = "Occupation Data + Job Zones (occupation_data.csv, job_zones.csv)"

LINK_SYSTEM_PROMPT = """Extract the occupations (jobs) that the user's message is about: the user's current or past job, and any target jobs they ask about. Write each as a standard English occupation title as used by O*NET (U.S. Department of Labor), singular or plural, e.g. "Telemarketers", "Registered Nurses", "Software Developers". Translate job names written in other languages into English. Do not invent jobs that are not mentioned or clearly implied.
Respond ONLY with JSON: {"occupations": ["<title>", ...]} - use [] if the message names no occupation."""

CONTEXT_SYSTEM_PROMPT = """You rewrite the user's latest chat message about occupations into a standalone question, using the conversation so far. Resolve references such as "it", "they", "those jobs", "the second one", "the less exposed ones" to the exact occupation titles mentioned before. Write the standalone question in English and keep the user's intent; do not answer the question.
Also list the occupations the standalone question is about, as standard English O*NET titles exactly as they appear in the conversation: the user's own job if it is still relevant, and every job the message refers to (at most 6). Use [] if no occupation is involved.
Respond ONLY with JSON: {"standalone_question": "<question>", "occupations": ["<title>", ...]}"""

# an occupation name counts as "close" to an O*NET occupation when it is a
# title / alternate title or a substring of one (search_occupations >= 90);
# fuzzy-only hits (<= 89) are not offered - the name is reported as missing
_CLOSE_MATCH_SCORE = 90


def _norm_text(s: str) -> str:
    s = re.sub(r"[^a-z0-9\- ]+", " ", (s or "").lower())
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def title_in_text(title: str, text: str) -> bool:
    """True when the whole title (singular or plural) is written in the text:
    "Registered Nurse" / "Registered Nurses" both count, "preschool teachers"
    is not "Preschool Teachers, Except Special Education"."""
    t = _norm_text(title).strip()
    if not t:
        return False
    words = t.split()
    last = words[-1]
    forms = {t, t + "s", t + "es"}
    if last.endswith("ies"):
        forms.add(" ".join(words[:-1] + [last[:-3] + "y"]))
    elif last.endswith("es") and len(last) > 4:
        forms.add(" ".join(words[:-1] + [last[:-2]]))
    if last.endswith("s") and len(last) > 3:
        forms.add(" ".join(words[:-1] + [last[:-1]]))
    q = _norm_text(text)
    return any(f" {f} " in q for f in forms)

# follow-up wording that points back into the conversation
_REFERENTIAL = re.compile(
    r"\b(it|its|they|them|their|those|these|this one|that one|the (first|second|third|"
    r"fourth|fifth|sixth|last|former|latter|other|same)|which of|among (them|those|these)|"
    r"above|previous|earlier|you (said|mentioned|recommended|listed)|what about|how about|"
    r"and the|more about|tell me more|why)\b")


# ------------------------------------------------------------------- engine
class ChatEngine:
    def __init__(self, context_budget: int = bootstrap.DEFAULT_CONTEXT_BUDGET):
        cfg = load_config()
        cfg["budget"]["context_tokens"] = int(context_budget)     # in memory only
        cfg["models"]["generator"] = "openrouter"                 # label only
        self.cfg = cfg
        self.budget = int(context_budget)
        self.G = store.load_graph(data_path(cfg, "graph", "meta.json").parent)
        # + O*NET occupations outside the research sample (Data Scientists, ...)
        self.graph_ext = graph_ext.merge_into(self.G)
        # Related Occupations ranks are whole numbers in O*NET; parquet stores
        # them as floats ("rank 2.0" in evidence lines) - restore the integer
        for _, _, d in self.G.edges(data=True):
            r = d.get("index")
            if d.get("type") == "RELATED_TO" and isinstance(r, float) and r == r:
                d["index"] = int(r)
        self.anchors = self._load_anchors(cfg)
        self.linker = EntityLinker(self.G)
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.default_llm = (OpenRouterLLM(key, bootstrap.OPENROUTER_DEFAULT_MODEL)
                            if key else _MissingKeyLLM())
        # kept for backward compatibility (engine.rag.budget / engine.rag.llm)
        self.rag = self._make_rag(self.default_llm, self.budget)
        self.edge_index = {k: (u, v, k, d)
                           for u, v, k, d in self.G.edges(keys=True, data=True)}
        self.occupations: list[tuple[str, str]] = sorted(
            (self.G.nodes[n]["name"], n) for n in self.G
            if self.G.nodes[n].get("label") == "Job")
        self._title_to_id = {t: n for t, n in self.occupations}
        # lowercase surface -> SOC for occupation search (titles + alt titles)
        self._occ_surfaces: list[tuple[str, str, str]] = []  # (surface_lower, soc, title)
        for title, nid in self.occupations:
            self._occ_surfaces.append((title.lower(), nid, title))
            for alt in self.G.nodes[nid].get("alt_titles") or []:
                self._occ_surfaces.append((str(alt).lower(), nid, title))
        self._qid = itertools.count(1)

    @staticmethod
    def _load_anchors(cfg: dict) -> LevelAnchors | None:
        """Behavioral level anchors: the bundled data/onet/ file first, then a
        full O*NET CSV folder if configured; None just drops the anchor
        phrases from RATED evidence lines."""
        bundled = bootstrap.DATA_DIR / "onet"
        for candidate in (bundled, cfg["onet"]["data_dir"]):
            try:
                return LevelAnchors(OnetData(candidate))
            except (FileNotFoundError, OSError, KeyError, ValueError):
                continue
        return None

    def _make_rag(self, llm, budget: int) -> HybridGraphRAG:
        rag = HybridGraphRAG(self.cfg, llm, self.G, self.anchors, self.linker)
        rag.budget = int(budget)
        return rag

    # ------------------------------------------------------------- backend
    @property
    def backend_description(self) -> str:
        return getattr(self.rag.llm, "description", type(self.rag.llm).__name__)

    @property
    def is_configured(self) -> bool:
        """True when a server-side default key exists (env / .env)."""
        return isinstance(self.default_llm, OpenRouterLLM)

    @property
    def default_model(self) -> str:
        return getattr(self.default_llm, "model", None) or bootstrap.OPENROUTER_DEFAULT_MODEL

    def set_backend(self, api_key: str, model: str | None = None) -> str:
        """Change the process-wide default OpenRouter key/model."""
        self.default_llm = OpenRouterLLM(api_key, model or bootstrap.OPENROUTER_DEFAULT_MODEL)
        self.rag = self._make_rag(self.default_llm, self.budget)
        self.cfg["models"]["generator"] = self.default_llm.model
        return self.backend_description

    def set_budget(self, tokens: int) -> None:
        self.budget = self._clamp_budget(tokens)
        self.rag.budget = self.budget

    @staticmethod
    def _clamp_budget(tokens: int | None) -> int:
        if tokens is None:
            return bootstrap.DEFAULT_CONTEXT_BUDGET
        return max(bootstrap.MIN_CONTEXT_BUDGET,
                   min(bootstrap.MAX_CONTEXT_BUDGET, int(tokens)))

    # ------------------------------------------------------------- lookups
    def graph_stats(self) -> dict:
        meta_path = bootstrap.DATA_DIR / "graph" / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        meta = graph_ext.merged_stats(meta, self.graph_ext)
        meta.setdefault("nodes_total", self.G.number_of_nodes())
        meta.setdefault("edges_total", self.G.number_of_edges())
        return meta

    def node_info(self, node_id: str) -> tuple[str, str]:
        a = self.G.nodes[node_id]
        return a.get("name", node_id), a.get("label", "?")

    def node_profile(self, node_id: str) -> dict | None:
        """All node attributes plus a by-type count of incident edges."""
        if node_id not in self.G:
            return None
        a = dict(self.G.nodes[node_id])
        out_counts: dict[str, int] = {}
        in_counts: dict[str, int] = {}
        for _, _, d in self.G.out_edges(node_id, data=True):
            out_counts[d.get("type", "?")] = out_counts.get(d.get("type", "?"), 0) + 1
        for _, _, d in self.G.in_edges(node_id, data=True):
            in_counts[d.get("type", "?")] = in_counts.get(d.get("type", "?"), 0) + 1
        return {"id": node_id, "label": a.pop("label", "?"), "name": a.pop("name", node_id),
                "attributes": _jsonable(a), "out_edges": out_counts, "in_edges": in_counts}

    def edge_record(self, edge_id: int, cited: bool = False) -> dict | None:
        e = self.edge_index.get(int(edge_id))
        if e is None:
            return None
        u, v, k, d = e
        return {"id": int(k), "src": u, "dst": v, "type": d.get("type"),
                "src_name": self.G.nodes[u].get("name", u),
                "dst_name": self.G.nodes[v].get("name", v),
                "statement": _strip_tag(f"E{k}", edge_line(self.G, e, self.anchors)),
                "attributes": _jsonable({kk: vv for kk, vv in d.items() if kk != "type"}),
                "cited": cited}

    def subgraph(self, edge_ids: list[int], cited_edge_ids: set[int] | None = None) -> dict:
        cited = set(cited_edge_ids or ())
        edges, nodes = [], {}
        for eid in edge_ids:
            rec = self.edge_record(eid, cited=int(eid) in cited)
            if rec is None:
                continue
            edges.append(rec)
            for n in (rec["src"], rec["dst"]):
                if n not in nodes:
                    name, label = self.node_info(n)
                    nodes[n] = {"id": n, "name": name, "label": label}
        return {"nodes": list(nodes.values()), "edges": edges}

    def ego_subgraph(self, center: str, radius: int = 1,
                     edge_types: set[str] | None = None, max_edges: int = 150) -> dict:
        if center not in self.G:
            raise KeyError(center)
        edges = q.khop_edges(self.G, [center], depth=max(1, min(3, int(radius))),
                             edge_types=edge_types, max_edges=max(1, min(1000, int(max_edges))))
        sg = self.subgraph([k for _, _, k, _ in edges])
        sg["center"] = center
        sg["total_incident_edges"] = self.G.degree(center)
        return sg

    def search_occupations(self, text: str, limit: int = 10) -> list[dict]:
        """Occupations whose title or alternate title matches `text`.

        Ranking: exact title (100) > exact alternate title (98) > title
        substring (95) > alternate-title substring (90) > fuzzy match
        (rapidfuzz WRatio, capped at 89). One row per occupation."""
        from rapidfuzz import fuzz, process

        t = (text or "").strip().lower()
        if not t:
            return []
        limit = max(1, min(100, int(limit)))
        best: dict[str, tuple[float, str, str]] = {}   # soc -> (score, via, title)

        def consider(soc: str, title: str, via: str, score: float):
            if score > best.get(soc, (0.0, "", ""))[0]:
                best[soc] = (score, via, title)

        for surf, soc, title in self._occ_surfaces:
            is_title = surf == title.lower()
            if surf == t:
                score = 100.0 if is_title else 98.0
            elif t in surf:
                score = 95.0 if is_title else 90.0
                if surf.startswith(t):
                    score += 1.0
            else:
                continue
            consider(soc, title, "title" if is_title else surf, score)
        if len(best) < limit:
            choices = {i: s for i, (s, _, _) in enumerate(self._occ_surfaces)}
            for surf, score, i in process.extract(t, choices, scorer=fuzz.WRatio,
                                                  limit=limit * 4, score_cutoff=70):
                _, soc, title = self._occ_surfaces[i]
                is_title = surf == title.lower()
                consider(soc, title, "title" if is_title else surf, min(89.0, float(score)))
        # ties: the more generic title (fewer words) first, then alphabetical
        ranked = sorted(best.items(),
                        key=lambda kv: (-kv[1][0], len(kv[1][2].split()), kv[1][2]))
        return [{"soc": soc, "title": title, "matched_on": via, "score": round(score, 1)}
                for soc, (score, via, title) in ranked[:limit]]

    # ------------------------------------------------------------- linking
    def preview_seeds(self, question: str) -> list[tuple[str, str, str]]:
        """(node_id, name, label) the exact entity linker finds."""
        out = []
        for n in self.linker.link(question):
            a = self.G.nodes[n]
            out.append((n, a.get("name", n), a.get("label", "?")))
        return out

    def resolve_seeds(self, question: str,
                      pinned: list[str] | tuple = (),
                      auto_match: bool = True) -> SeedResolution:
        """Pins first, then the exact linker, then fuzzy occupation matches
        for question words neither of those covered."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for sid in pinned or ():
            if sid in self.G and sid not in seen:
                seen.add(sid)
                name, label = self.node_info(sid)
                entities.append(Entity(sid, name, label, "pinned"))
        linked = [(nid, name, label) for nid, name, label in self.preview_seeds(question)
                  if not self._weak_link(nid, label, question)]
        for nid, name, label in linked:
            if nid not in seen:
                seen.add(nid)
                entities.append(Entity(nid, name, label, "linked"))
        suggestions: dict[str, list[tuple[str, str]]] = {}
        if auto_match:
            linked_jobs = [nid for nid, _, label in linked if label == "Job"]
            exclude = list(pinned or ()) + linked_jobs
            detailed = self.suggest_occupations_detailed(question)
            suggestions = {t: info["candidates"][:5] for t, info in detailed.items()}
            for nid, title, term, conf in self._pick_auto(detailed, exclude):
                if nid not in seen:
                    seen.add(nid)
                    entities.append(Entity(nid, title, "Job", "matched", term, conf))
        # retrieval seeds: pins + matches are forced; linked ones the linker
        # re-finds itself (HybridGraphRAG.seeds_for unions both).
        seeds = [e.node_id for e in entities if e.source != "linked"]
        return SeedResolution(entities=entities, seeds=seeds, suggestions=suggestions)

    def propose_occupations(self, question: str, *, history: list[dict] | None = None,
                            api_key: str | None = None, model: str | None = None,
                            auto_match: bool = True) -> dict:
        """The confirmation step before a question is answered.

        Returns {"confirm": bool, "terms": [{"term", "candidates": [(soc, title)],
        "via"}], "entities": [Entity]}. confirm is True when the question
        names no occupation that exists in O*NET under that exact title (or
        alternate title), yet close occupations were found - from the wording
        ("professor" -> the Postsecondary Teachers) or, when the wording
        names none, from the model's reading of the message ("I sell things
        over the phone" -> Telemarketers). The UI then asks the user which of
        the candidates to proceed with instead of guessing. No confirmation
        for questions that name an occupation exactly, name none (skill or
        task questions), or point back into the conversation ("those jobs").
        """
        res = self.resolve_seeds(question, auto_match=auto_match)
        out = {"confirm": False, "terms": [], "entities": res.entities}
        history = memory.normalize_turns(history)
        if history and _REFERENTIAL.search(question.lower()):
            return out                                  # a follow-up: memory resolves it
        # An occupation needs no confirmation only when its full O*NET title
        # is written in the question. Everything else is confirmed, each
        # name on its own: "zookeepers and preschool teachers" asks about
        # both (Animal Caretakers; Preschool Teachers, Except Special
        # Education), "Software Developers and Data Scientists" about neither.
        exact = {e.node_id for e in res.entities
                 if e.label == "Job" and title_in_text(e.name, question)}
        terms: list[dict] = []
        covered: set[str] = set(exact)
        if res.suggestions:
            for term, info in self.suggest_occupations_detailed(question).items():
                cands = [(s_, t) for s_, t in info["candidates"] if s_ not in exact][:6]
                if cands:
                    terms.append({"term": term, "candidates": cands, "via": "wording"})
                    covered |= {s_ for s_, _ in cands}
        for e in res.entities:          # placed by the exact linker through an alternate title
            if e.label != "Job" or e.node_id in covered or e.source not in ("linked", "matched"):
                continue
            wording = e.term or self._wording_for(e.node_id, question) or e.name
            cands = [(e.node_id, e.name)] + [
                (h["soc"], h["title"]) for h in self._close_occupations(wording)
                if h["soc"] != e.node_id and h["soc"] not in exact][:3]
            terms.append({"term": wording, "candidates": cands, "via": "wording"})
            covered |= {s_ for s_, _ in cands}
        # The model's reading of the message catches occupations the wording
        # matcher could not place at all ("astronaut", "I sell things over
        # the phone"). Names it reads that match nothing close in O*NET are
        # reported as missing, so the user can re-check the name first.
        if api_key or not isinstance(self.default_llm, _MissingKeyLLM):
            llm = (OpenRouterLLM(api_key, model or self.default_model) if api_key
                   else self.default_llm)
            for name in self._llm_occupation_names(question, llm):
                hits = self._close_occupations(name)
                if hits and hits[0]["soc"] in covered:
                    continue
                best = hits[0]["score"] if hits else 0
                if best >= _CLOSE_MATCH_SCORE:
                    cands = [(h["soc"], h["title"]) for h in hits
                             if h["score"] >= _CLOSE_MATCH_SCORE and h["soc"] not in exact]
                    terms.append({"term": name, "candidates": cands, "via": "model"})
                elif not any(t["term"].lower() == name.lower() for t in terms):
                    terms.append({"term": name, "candidates": [], "via": "model"})
        seen: set[str] = set()
        for t in terms:                                 # one card per occupation
            t["candidates"] = [(s_, n) for s_, n in t["candidates"]
                               if not (s_ in seen or seen.add(s_))]
        terms = [t for t in terms if t["candidates"] or t.get("via") == "model"][:4]
        out.update(confirm=bool(terms), terms=terms)
        return out

    def _wording_for(self, nid: str, text: str) -> str | None:
        """The alternate title of occupation `nid` that is written in the
        text ("zookeeper" for Animal Caretakers), longest first; None if
        the text names it some other way."""
        alts = [str(a) for a in self.G.nodes[nid].get("alt_titles") or []]
        for alt in sorted(alts, key=len, reverse=True):
            if title_in_text(alt, text):
                return alt.lower()
        return None

    def _close_occupations(self, name: str, limit: int = 4) -> list[dict]:
        """search_occupations over the name and its singular form ("zookeepers"
        -> "zookeeper" is an alternate title), best score per occupation."""
        forms = [name]
        low = name.strip().lower()
        for suffix in ("ies", "es", "s"):
            if low.endswith(suffix) and len(low) > len(suffix) + 2:
                forms.append(low[:-len(suffix)] + ("y" if suffix == "ies" else ""))
                break
        best: dict[str, dict] = {}
        for form in forms:
            for h in self.search_occupations(form, limit * 2):
                if h["score"] > best.get(h["soc"], {"score": 0})["score"]:
                    best[h["soc"]] = h
        return sorted(best.values(), key=lambda h: -h["score"])[:limit]

    def _llm_occupation_names(self, question: str, llm) -> list[str]:
        """Occupation titles the model reads in the message (best effort)."""
        try:
            resp = llm.complete(model=self.cfg["models"]["generator"],
                                system=LINK_SYSTEM_PROMPT, user=question, max_tokens=200)
            obj = self._parse_json(resp.get("text", "")) or {}
            names = obj.get("occupations") or []
        except Exception:  # noqa: BLE001 - linking help is best-effort
            return []
        return [n.strip() for n in names[:4] if isinstance(n, str) and n.strip()]

    def _weak_link(self, nid: str, label: str, question: str) -> bool:
        """True when a Job was linked only through a single-word alternate
        title written in lowercase prose ("marketer" inside "tele marketer"
        -> Market Research Analysts). Such links are dropped; the fuzzy
        matcher re-finds the right occupation from the whole word."""
        if label != "Job":
            return False
        a = self.G.nodes[nid]
        ql = question.lower()
        if str(a.get("name", "")).lower() in ql or nid in question:
            return False
        for alt in a.get("alt_titles") or []:
            alt_l = str(alt).lower()
            if " " in alt_l or len(alt_l) < 4:
                continue
            for m in re.finditer(rf"\b{re.escape(alt_l)}\b", ql):
                if question[m.start():m.end()] != alt_l:   # capitalised: intended
                    return False
        # name absent, every matching alt title is a lowercase single word
        return True

    # ------------------------------------------------ LLM-assisted linking
    # ------------------------------------------------- conversation memory
    @staticmethod
    def _needs_context(question: str, entities: list[Entity]) -> bool:
        """A follow-up must be read with the history when it names no
        confident occupation or points back into the conversation."""
        confident = any(e.label == "Job" and e.confidence == "high" for e in entities)
        ql = question.lower()
        referential = bool(_REFERENTIAL.search(ql))
        return not confident or referential

    def _contextualize(self, question: str, history: list[dict], llm) -> str | None:
        """Ask the model to rewrite the follow-up as a standalone question.
        The rewrite is shown to the model with the question (its reading of
        the conversation) and reported as `standalone_question`; retrieval
        never uses it, so the evidence does not depend on the model."""
        block = HybridGraphRAG.history_block(history)
        try:
            resp = llm.complete(model=self.cfg["models"]["generator"],
                                system=CONTEXT_SYSTEM_PROMPT,
                                user=f"{block}\n\nLatest message: {question}", max_tokens=300)
            obj = self._parse_json(resp.get("text", "")) or {}
        except Exception:  # noqa: BLE001 - memory help is best-effort
            return None
        standalone = obj.get("standalone_question")
        return standalone.strip() if isinstance(standalone, str) and standalone.strip() else None

    def _carry_over(self, history: list[dict], exclude: set[str]) -> list[Entity]:
        """The occupations in focus in the most recent turn that had any:
        its seeds, then the occupations its answer listed (exact O*NET
        titles). Deterministic, so a follow-up retrieves the same records
        whichever model answers."""
        for turn in reversed(history):
            out: list[Entity] = []
            socs = [s for s in (turn.get("seeds") or []) if s in self.G]
            for title in turn.get("occupations") or []:
                hits = self.search_occupations(str(title), 1)
                if hits and hits[0]["score"] >= 100 and hits[0]["soc"] not in socs:
                    socs.append(hits[0]["soc"])
            for soc in socs:
                if soc not in exclude and self.G.nodes[soc].get("label") == "Job":
                    exclude.add(soc)
                    out.append(Entity(soc, self.G.nodes[soc]["name"], "Job", "memory"))
            if out:
                return out[:6]
        return []

    # ---------------------------------------------------- fuzzy job matching
    _STOP = frozenset("""a an the and or of for to in on at by with from about
        how what which who whom whose why when where is are was were be been
        do does did can could will would should may might their his her its
        they them then than he she it my your our this that these those there
        here also very much many more most some any all both each other same
        different job jobs occupation occupations profession professions work
        works working career careers role roles position positions related
        relate relates relationship relationships compare compared comparison
        common share shared task tasks skill skills ability abilities
        knowledge activity activities important require requires
        requirement requirements need needs people person someone
        matter matters perform performs performed performing day days way
        ways thing things kind kinds type types good best better worst look
        looks make makes making use used uses using help helps get gets lot
        similar overlap overlaps differ differs difference differences
        between versus every daily technology technologies
        zone zones level levels importance rating ratings rate rates
        description descriptions code codes title titles alternate example
        examples list lists name names number numbers percent average high
        low top bottom main major core supplemental hot demand detailed
        intermediate general essential transferable style styles tell explain
        describe show give over under into onto next after before during
        through without within across around along above below down back
        away off out up like want wants wanted wanting think know feel love
        enjoy hate something anything nothing everything well still just only
        even now today tomorrow year years month months week weeks time times
        life living money pay salary home house office phone online remote
        part full hour hours long short hard easy new old young current
        currently former future past please thanks hello sell sells selling
        buy work worked change switch move become start stop leave quit
        find found looking seek seeking apply interested interest""".split())

    @staticmethod
    def _normalize_word(w: str) -> str:
        w = w.removesuffix("'s").rstrip("'")
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # naive singular: teachers -> teacher
        return w

    def _question_ngrams(self, question: str) -> tuple[list[str], list[str]]:
        """(bigrams, unigrams) of normalized content words, in question order.
        Bigrams catch multi-word occupations ("data analyst", "air traffic")."""
        raw = [self._normalize_word(w)
               for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", question.lower())]

        def stopped(w: str) -> bool:
            # check the singularized form AND its plural, since naive
            # singularization can mangle stopwords (activities -> activitie)
            return w in self._STOP or w + "s" in self._STOP

        bigrams = [f"{a} {b}" for a, b in zip(raw, raw[1:])
                   if len(a) >= 3 and len(b) >= 3
                   and not (stopped(a) or stopped(b))]
        seen: set[str] = set()
        unigrams = [w for w in raw
                    if len(w) >= 4 and not stopped(w)
                    and not (w in seen or seen.add(w))]
        return bigrams, unigrams

    def _question_trigrams(self, question: str) -> list[tuple[int, list[str]]]:
        """Adjacent word triples that could be a three-word occupation title,
        as (start position, [phrase as written, singularized phrase]). The
        first word may be a common word ("high school teacher"); the other
        two must be content words."""
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", question.lower())
        norm = [self._normalize_word(w) for w in words]

        def stopped(w: str) -> bool:
            return w in self._STOP or w + "s" in self._STOP

        out = []
        for i in range(len(words) - 2):
            a, b, c = norm[i:i + 3]
            if (len(a) >= 3 and len(b) >= 3 and len(c) >= 3
                    and a not in ("the", "and", "for", "with", "from", "about")
                    and not stopped(b) and not stopped(c)):
                # "computer systems analyst": keep inner plurals as written
                written = f"{words[i]} {words[i + 1]} {c}"
                out.append((i, list(dict.fromkeys([written, f"{a} {b} {c}"]))))
        return out

    def _match_term(self, term: str, head_only: bool) -> list[tuple[str, str]]:
        return self._match_term_detail(term, head_only)[0]

    def _match_term_detail(self, term: str, head_only: bool
                           ) -> tuple[list[tuple[str, str]], str]:
        """Occupations whose title (else alternate title) contains `term` as
        whole words, most generic first: ([(node_id, title), ...], via).

        head_only (used for single words) requires the term to be the head
        noun of the title — "Registered Nurses" for "nurse" — so ordinary
        words that merely occur inside a title or alternate title ("zone" in
        "Zone Manager", "data" in "Data Entry Keyers") never pull in an
        unrelated occupation. Trailing qualifiers after a comma are allowed
        ("Bus Drivers, School", "Managers, All Other")."""
        tail = r"(?:,.*)?$" if head_only else ""
        pat = re.compile(rf"\b{re.escape(term)}s?\b{tail}")
        hits = [(title, nid) for title, nid in self.occupations
                if pat.search(title.lower())]
        if hits:
            # fewer words, then shorter = more generic title first
            hits.sort(key=lambda t: (len(t[0].split()), len(t[0]), t[0]))
            return [(nid, title) for title, nid in hits], "title"
        scored = []  # alternate-title fallback
        for title, nid in self.occupations:
            matches = [str(alt).lower() for alt in self.G.nodes[nid].get("alt_titles") or []
                       if pat.search(str(alt).lower())]
            if matches:
                exact = any(al in (term, term + "s") for al in matches)
                starts = any(al.startswith(term) for al in matches)
                # exact alt title, then number of matching alt titles (an
                # occupation known by many "<x> doctor" titles is the better
                # guess), then the shortest matching alt title
                scored.append(((0 if exact else 1 if starts else 2),
                               -len(matches), min(len(al) for al in matches),
                               title, nid))
        scored.sort()
        return [(nid, title) for *_, title, nid in scored], ("alt" if scored else "none")

    def suggest_occupations(self, question: str,
                            per_term: int = 5) -> dict[str, list[tuple[str, str]]]:
        """{question term: [(node_id, title), ...]} for every content word or
        word pair that names an occupation in the 923 titles or their
        alternate titles. A matched bigram absorbs its unigrams."""
        return {term: info["candidates"][:per_term]
                for term, info in self.suggest_occupations_detailed(question).items()}

    def suggest_occupations_detailed(self, question: str) -> dict[str, dict]:
        """Like suggest_occupations, plus how each term matched:
        {"candidates": [...], "via": "title"|"alt"|"fuzzy", "ngram": "bigram"|"unigram",
         "confidence": "high"|"low"}. A single word that only hit an alternate
        title is low confidence (alt titles contain many ordinary words)."""
        bigrams, unigrams = self._question_ngrams(question)
        out: dict[str, dict] = {}
        used_words: set[str] = set()
        # three-word titles first ("elementary school teacher", "high school
        # teacher"): otherwise the overlapping pairs ("elementary school",
        # "school teacher") each pick an occupation, and a common first word
        # ("high") is dropped. Matched word POSITIONS are consumed, so
        # "heavy truck driver vs bus driver" still finds "bus driver".
        used_pos: set[int] = set()
        for pos, variants in self._question_trigrams(question):
            if {pos, pos + 1, pos + 2} & used_pos:
                continue
            for tg in variants:
                m, via = self._match_term_detail(tg, head_only=False)
                if m and via == "alt":
                    # an alternate title must be the whole phrase, not merely
                    # contain it ("high school teacher" = Secondary School Teachers)
                    m = [(n, t) for n, t in m
                         if any(str(al).lower() in (tg, tg + "s")
                                for al in self.G.nodes[n].get("alt_titles") or [])]
                if m and via in ("title", "alt"):
                    out[tg] = {"candidates": m, "via": via, "ngram": "trigram",
                               "confidence": "high"}
                    used_pos |= {pos, pos + 1, pos + 2}
                    break
        if used_pos:
            norm = [self._normalize_word(w)
                    for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", question.lower())]
            free_pairs = {f"{norm[i]} {norm[i + 1]}" for i in range(len(norm) - 1)
                          if not {i, i + 1} & used_pos}
            free_words = {norm[i] for i in range(len(norm)) if i not in used_pos}
            bigrams = [bg for bg in bigrams if bg in free_pairs]
            unigrams = [w for w in unigrams if w in free_words]
        for bg in bigrams:
            m, via = self._match_term_detail(bg, head_only=False)
            if not m:   # "tele marketer" -> "telemarketer"; "co pilot" -> "co-pilot"
                for joined in (bg.replace(" ", ""), bg.replace(" ", "-")):
                    m, via = self._match_term_detail(joined, head_only=True)
                    if m:
                        break
            if not m:
                m, via = self._fuzzy_title(bg), "fuzzy"
            if m:
                out[bg] = {"candidates": m, "via": via, "ngram": "bigram",
                           "confidence": "high"}
                used_words |= set(bg.split())
        for term in unigrams:
            if term in used_words:
                continue
            m, via = self._match_term_detail(term, head_only=True)
            if not m and len(term) >= 6:
                m, via = self._fuzzy_title(term, cutoff=90), "fuzzy"   # typos
            if m:
                out[term] = {"candidates": m, "via": via, "ngram": "unigram",
                             "confidence": "high" if via == "title" else "low"}
        return out

    def _fuzzy_title(self, term: str, cutoff: int = 88) -> list[tuple[str, str]]:
        """Whole-string similarity against primary titles (singular/plural
        tolerant). Safe for ordinary words: "zone" vs any title scores low."""
        from rapidfuzz import fuzz, process

        choices = {nid: title.lower() for title, nid in self.occupations}
        hits = process.extract(term, choices, scorer=fuzz.ratio, limit=5,
                               score_cutoff=cutoff)
        if not hits:
            hits = process.extract(term + "s", choices, scorer=fuzz.ratio, limit=5,
                                   score_cutoff=cutoff)
        return [(nid, self.G.nodes[nid]["name"]) for _, _, nid in hits]

    def _pick_auto(self, detailed: dict[str, dict], exclude: list[str] | tuple,
                   max_jobs: int = 3) -> list[tuple[str, str, str, str]]:
        """[(node_id, title, term, confidence)] — best candidate per term."""
        excluded_titles = " | ".join(
            self.G.nodes[n].get("name", "").lower() for n in exclude if n in self.G)
        out: list[tuple[str, str, str, str]] = []
        seen: set[str] = set(exclude)
        for term, info in detailed.items():
            if excluded_titles and re.search(rf"\b{re.escape(term)}s?\b", excluded_titles):
                continue
            nid, title = info["candidates"][0]
            if nid not in seen:
                seen.add(nid)
                out.append((nid, title, term, info["confidence"]))
            if len(out) >= max_jobs:
                break
        return out

    def auto_occupations(self, question: str, max_jobs: int = 3,
                         exclude: list[str] | tuple = ()) -> list[tuple[str, str]]:
        """Best-guess occupation per question term: [(node_id, title), ...].
        Terms already covered by an `exclude` job's title are skipped."""
        picks = self._pick_auto(self.suggest_occupations_detailed(question), exclude, max_jobs)
        return [(nid, title) for nid, title, _, _ in picks]

    # ------------------------------------------------------------ questions
    def ask(self, question: str, forced_seeds: list[str] | None = None,
            qid: str | None = None, *, api_key: str | None = None,
            model: str | None = None, budget: int | None = None,
            auto_link: bool = True, history: list[dict] | None = None) -> Answer:
        """Answer one question.

        forced_seeds: node ids to pin (always seed retrieval).
        auto_link:    also add fuzzy occupation matches for words the exact
                      linker missed (what the UI does by default).
        api_key/model/budget: per-request overrides; otherwise the engine's
                      default key (env), default model and budget are used.
        history:      prior turns (memory.turn_from_answer records, oldest
                      first). A follow-up that names no occupation or refers
                      back ("those jobs") inherits the occupations in focus
                      from the last turn; the model's rewrite of it is shown
                      to the model as its reading only.

        Retrieval never depends on the language model: the seeds come from
        the wording (exact / alternate titles, confirmed choices, pins) or
        from the previous turn, and the evidence is cut from the graph with
        the user's own question. Whichever model answers, the same records
        are retrieved; the model only writes from them and cites them.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")
        if api_key:
            llm = OpenRouterLLM(api_key, model or self.default_model)
        elif model and isinstance(self.default_llm, OpenRouterLLM):
            llm = OpenRouterLLM(self.default_llm.api_key, model)
        else:
            llm = self.default_llm
        if isinstance(llm, _MissingKeyLLM):
            raise NotConfiguredError(
                "No OpenRouter API key is set. Pass one per request, or put "
                "OPENROUTER_API_KEY in Chatbot/.env.")
        rag = self._make_rag(llm, self._clamp_budget(budget) if budget else self.budget)

        res = self.resolve_seeds(question, pinned=forced_seeds or (), auto_match=auto_link)
        if forced_seeds:
            # the caller (or the user, through the confirmation step) named
            # the occupations: single-word alternate-title guesses are dropped
            res.entities = [e for e in res.entities
                            if not (e.source == "matched" and e.confidence == "low")]
            res.seeds = [e.node_id for e in res.entities if e.source != "linked"]
        history = memory.normalize_turns(history)[-HISTORY_TURNS:]
        standalone = None
        if history and self._needs_context(question, res.entities):
            # The occupations in focus carry over from the last turn (no
            # model involved). The model's rewrite of the follow-up is only
            # shown to it with the question; retrieval never sees it.
            known = {e.node_id for e in res.entities}
            standalone = self._contextualize(question, history, llm)
            contextual = self._carry_over(history, known)
            if contextual:
                res.entities = [e for e in res.entities
                                if not (e.source == "matched" and e.confidence == "low")
                                ] + contextual
                res.seeds = [e.node_id for e in res.entities if e.source != "linked"]
        aliases = []
        for e in res.entities:
            if e.label != "Job" or title_in_text(e.name, question):
                continue
            wording = e.term or self._wording_for(e.node_id, question)
            if wording:
                aliases.append(f"'{wording}' = {e.name} ({e.node_id})")
        item = {"qid": qid or f"chat-{next(self._qid)}-{time.time():.0f}",
                "question_template": question, "question_interpreted": standalone,
                "history": history, "aliases": aliases,
                "forced_seeds": res.seeds,
                # the fully resolved entity list (pins, filtered exact links,
                # fuzzy matches, inferred) is the seed set; the research
                # linker is not re-run inside retrieval
                "seeds": [e.node_id for e in res.entities]}
        rec = rag.answer(item)

        parsed = self._parse_json(rec["raw"])
        if parsed is not None and (rec.get("usage") or {}).get("finish_reason") == "length":
            parsed["_truncated"] = True   # provider says the reply hit max_tokens
        unit_ids: list[str] = list(rec["context_units"])
        # the citations array plus every id named inline in the structured
        # fields (a truncated reply may lose the array but keep inline ids)
        citations, invalid = self._normalize_citations(
            self._all_citation_tokens(parsed), unit_ids)
        cited_units = set(citations)
        cited_edges = sorted({e for uid in cited_units
                              for e in rec["unit_edge_ids"].get(uid, [])})

        texts = self._unit_texts(rec["context_text"])
        evidence = [(uid, texts.get(uid, ""), uid in cited_units) for uid in unit_ids]
        # retrieval seeds = the resolved entities (profiles of related
        # occupations may also enter the evidence, but they are not seeds)
        seeds = [n for n in item["seeds"] if n in self.G]

        if parsed is None:
            status = "unparsed"
        elif parsed.get("_truncated"):
            status = "truncated"
        elif citations and not invalid:
            status = "grounded"
        elif citations:
            status = "partially_grounded"
        else:
            status = "ungrounded"

        sources = [self.source_record(uid, texts.get(uid, "")) for uid in citations]
        ai_relevant = (ai_question(standalone or question)
                       or any(c.startswith("X-") for c in citations))
        seed_jobs = [n for n in seeds if self.G.nodes[n].get("label") == "Job"]
        cards = self._build_cards(parsed, unit_ids, rec, seed_jobs, cited_units)
        exposure = self._exposure_rows(seed_jobs, cards)
        focus_edges, focus_nodes = self._focus_graph(
            cited_edges, seed_jobs, cards, citations, list(rec["context_edge_ids"]))
        return Answer(question=question, parsed=parsed, raw=rec["raw"],
                      seeds=seeds, evidence=evidence,
                      cited_edge_ids=cited_edges,
                      retrieved_edge_ids=list(rec["context_edge_ids"]),
                      usage=rec["usage"], context_tokens=rec["context_tokens"],
                      citations=citations, invalid_citations=invalid,
                      status=status, model=rec.get("model"),
                      entities=res.entities, sources=sources,
                      headline=self._headline(parsed),
                      key_points=self._str_list(parsed, "key_points", 5),
                      occupations=cards,
                      follow_ups=self._str_list(parsed, "follow_ups", 3),
                      ai_exposure=exposure,
                      focus_edge_ids=focus_edges, focus_node_ids=focus_nodes,
                      standalone_question=standalone, history_turns=len(history),
                      ai_relevant=ai_relevant, rec=rec)

    # ------------------------------------------------- structured answer
    @staticmethod
    def _all_citation_tokens(parsed: dict | None) -> list | None:
        if parsed is None:
            return None
        raw = parsed.get("citations")
        tokens = list(raw) if isinstance(raw, list) else ([raw] if raw else [])
        texts = [parsed.get("headline"), parsed.get("answer")]
        for k in ("key_points",):
            if isinstance(parsed.get(k), list):
                texts += parsed[k]
        for occ in parsed.get("occupations") or []:
            if isinstance(occ, dict):
                texts.append(occ.get("fit"))
                c = occ.get("citations")
                tokens += c if isinstance(c, list) else ([c] if c else [])
        for t in texts:
            if isinstance(t, list):
                t = " ".join(map(str, t))
            if isinstance(t, str) and t:
                tokens += ChatEngine._CIT_RE.findall(t)
        return tokens

    @staticmethod
    def _str_list(parsed: dict | None, key: str, limit: int) -> list[str]:
        v = (parsed or {}).get(key)
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if isinstance(x, (str, int, float))
                and str(x).strip()][:limit]

    @staticmethod
    def _headline(parsed: dict | None) -> str:
        if not parsed:
            return ""
        h = parsed.get("headline")
        if isinstance(h, str) and h.strip():
            return h.strip()
        a = parsed.get("answer")
        if isinstance(a, str):
            text = re.sub(r"[#*_`>]+", "", a).strip()
            first = re.split(r"(?<=[.!?。])\s|\n", text, maxsplit=1)[0].strip()
            return first[:220]
        if isinstance(a, bool):
            return "Yes" if a else "No"
        if isinstance(a, (int, float)):
            return str(a)
        if isinstance(a, list):
            return f"{len(a)} item(s): " + ", ".join(map(str, a[:3])) + (" …" if len(a) > 3 else "")
        return ""

    def _evidence_jobs(self, rec: dict) -> list[str]:
        """Occupations that appear in the evidence the model saw, in order."""
        out: list[str] = []
        for uid in rec["context_units"]:
            nodes = rec.get("unit_node_ids", {}).get(uid) or []
            if uid.startswith(("N-", "X-")):
                nodes = [uid[2:]]
            for n in nodes:
                if n in self.G and self.G.nodes[n].get("label") == "Job" and n not in out:
                    out.append(n)
        return out

    def _resolve_job(self, soc, title) -> str | None:
        if isinstance(soc, str) and soc.strip() in self.G and \
                self.G.nodes[soc.strip()].get("label") == "Job":
            return soc.strip()
        if isinstance(title, str) and title.strip():
            t = title.strip()
            if t in self._title_to_id:
                return self._title_to_id[t]
            hits = self.search_occupations(t, 1)
            if hits and hits[0]["score"] >= 95:
                return hits[0]["soc"]
        return None

    def _build_cards(self, parsed, unit_ids, rec, seed_jobs, cited_units) -> list[dict]:
        """Occupation cards for the occupations the model recommended or
        compared. Each must be an occupation the evidence actually contains;
        everything except the model's short "fit" text comes from the graph."""
        in_evidence = self._evidence_jobs(rec)
        allowed = set(in_evidence)
        ref = seed_jobs[0] if len(seed_jobs) == 1 else None
        cards, seen = [], set()
        items = (parsed or {}).get("occupations")
        for it in items if isinstance(items, list) else []:
            if isinstance(it, str):
                it = {"title": it}
            if not isinstance(it, dict):
                continue
            soc = self._resolve_job(it.get("soc"), it.get("title"))
            if soc is None or soc not in allowed or soc in seen:
                continue
            seen.add(soc)
            raw_c = it.get("citations")
            raw_c = raw_c if isinstance(raw_c, list) else ([raw_c] if raw_c else [])
            fit = it.get("fit") if isinstance(it.get("fit"), str) else ""
            valid, _ = self._normalize_citations(raw_c + self._CIT_RE.findall(fit), unit_ids)
            card = self.occupation_card(soc, ref=ref if soc != ref else None)
            card.update(fit=fit.strip(), citations=valid, source="model")
            cards.append(card)
            if len(cards) >= 6:
                break
        if not cards and ref is not None:
            # the model recommended in prose only: fall back to the related
            # occupations whose records it cited, in relatedness order
            for uid in unit_ids:
                if uid not in cited_units or not uid.startswith("E"):
                    continue
                e = self.edge_index.get(int(uid[1:])) if uid[1:].isdigit() else None
                if e and e[3].get("type") == "RELATED_TO" and e[0] == ref and e[1] not in seen:
                    seen.add(e[1])
                    card = self.occupation_card(e[1], ref=ref)
                    card.update(fit="", citations=[uid], source="cited related occupation")
                    cards.append(card)
            cards.sort(key=lambda c: c.get("related_rank") or 999)
            cards = cards[:6]
        if len(seed_jobs) > 1:
            # a comparison: every occupation asked about gets a card, in the
            # order asked, whether or not the model listed it
            for sj in seed_jobs:
                if sj not in seen and sj in allowed:
                    seen.add(sj)
                    card = self.occupation_card(sj)
                    card.update(fit="", citations=[u for u in (f"N-{sj}", f"X-{sj}")
                                                   if u in cited_units],
                                source="compared occupation")
                    cards.append(card)
            order = {s: i for i, s in enumerate(seed_jobs)}
            cards.sort(key=lambda c: order.get(c["soc"], len(order)))
            cards = cards[:6]
        return cards

    def occupation_card(self, soc: str, ref: str | None = None) -> dict:
        """Graph facts for one occupation (+ its relation to `ref`)."""
        a = self.G.nodes[soc]
        desc = str(a.get("description") or "")
        first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
        jz = a.get("job_zone")
        card = {"soc": soc, "title": a.get("name", soc),
                "job_zone": int(jz) if jz is not None and jz == jz else None,
                "description": first[:260],
                "related_rank": None, "related_tier": None, "related_to": None,
                "top_skills": [], "ai_exposure": None}
        if ref and ref in self.G:
            for _, v, _, d in q.out_edges(self.G, ref, "RELATED_TO"):
                if v == soc:
                    idx = d.get("index")
                    card.update(related_rank=int(idx) if idx is not None and idx == idx else None,
                                related_tier=d.get("tier"), related_to=ref)
                    break
        rag = self.rag
        card["top_skills"] = [
            {"name": self.G.nodes[v].get("name", v),
             "importance": d.get("importance"), "level": d.get("level")}
            for _, v, _, d in rag._top_rated_edges(soc, "TransferableSkill", 3)]
        card["ai_exposure"] = self.exposure_for(soc)
        return card

    def exposure_for(self, soc: str) -> dict | None:
        if soc not in self.G:
            return None
        x = ai_exposure.default_index().lookup(soc, self.G.nodes[soc].get("name", ""))
        return x.to_dict() if x else None

    def _exposure_rows(self, seed_jobs: list[str], cards: list[dict]) -> list[dict]:
        rows, seen = [], set()
        for role, socs in (("current", seed_jobs), ("recommended", [c["soc"] for c in cards])):
            for soc in socs:
                if soc in seen or soc not in self.G:
                    continue
                seen.add(soc)
                rows.append({"soc": soc, "title": self.G.nodes[soc].get("name", soc),
                             "role": role if len(seed_jobs) == 1 or role == "recommended"
                             else "discussed",
                             "exposure": self.exposure_for(soc)})
        return rows

    def _focus_graph(self, cited_edges: list[int], seed_jobs: list[str],
                     cards: list[dict], citations: list[str],
                     retrieved_edges: list[int] = ()) -> tuple[list[int], list[str]]:
        """The knowledge graph of this question: every retrieved O*NET record
        that touches an occupation the question is about. Retrieval runs
        before the language model and does not depend on it, so this graph
        is the same whichever model writes the answer; the model's citations
        are only highlighted on it. Added so that every card hangs off the
        graph: the cited edges, the Related Occupations edges from the
        current job to each recommended one, and every occupation whose
        profile or AI-exposure line was cited."""
        focus = set(seed_jobs)
        edges = [int(e) for e in retrieved_edges
                 if int(e) in self.edge_index
                 and (self.edge_index[int(e)][0] in focus or self.edge_index[int(e)][1] in focus)]
        have = set(edges)
        for e in cited_edges:
            if int(e) not in have:
                have.add(int(e))
                edges.append(int(e))
        for c in cards:
            for sj in seed_jobs:
                for _, v, k, d in q.out_edges(self.G, sj, "RELATED_TO"):
                    if v == c["soc"] and int(k) not in have:
                        have.add(int(k))
                        edges.append(int(k))
        nodes: list[str] = []
        for n in seed_jobs + [c["soc"] for c in cards] + \
                [u[2:] for u in citations if u.startswith(("N-", "X-"))]:
            if n in self.G and n not in nodes:
                nodes.append(n)
        return edges, nodes

    def source_record(self, unit_id: str, text: str = "") -> dict:
        """Human-readable provenance for one evidence id: the O*NET 30.3
        table the record comes from, the record statement, and its nodes."""
        base = {"id": unit_id, "statement": _strip_tag(unit_id, text), "dataset": "O*NET 30.3"}
        if unit_id.startswith("X-"):
            soc = unit_id[2:]
            name, label = self.node_info(soc) if soc in self.G else (soc, "?")
            x = ai_exposure.default_index().lookup(soc, name)
            if not base["statement"] and x is not None:
                base["statement"] = x.evidence_text(name)
            return {**base, "dataset": ai_exposure.SOURCE_NAME, "kind": "AI exposure",
                    "table": ai_exposure.SOURCE_CITATION, "url": ai_exposure.SOURCE_URL,
                    "nodes": [{"id": soc, "name": name, "label": label}]}
        if unit_id.startswith("N-"):
            nid = unit_id[2:]
            name, label = self.node_info(nid) if nid in self.G else (nid, "?")
            return {**base, "kind": "occupation profile", "table": _PROFILE_TABLE,
                    "nodes": [{"id": nid, "name": name, "label": label}]}
        if unit_id.startswith("E") and unit_id[1:].isdigit():
            e = self.edge_index.get(int(unit_id[1:]))
            if e is not None:
                u, v, k, d = e
                etype = d.get("type", "?")
                vl = self.G.nodes[v].get("label", "")
                table = (_RATED_TABLE.get(vl, "Descriptor ratings") if etype == "RATED"
                         else _ONET_TABLE.get(etype, etype))
                if not base["statement"]:
                    base["statement"] = _strip_tag(unit_id, edge_line(self.G, e, self.anchors))
                return {**base, "kind": etype, "table": table,
                        "nodes": [{"id": u, "name": self.G.nodes[u].get("name", u),
                                   "label": self.G.nodes[u].get("label", "")},
                                  {"id": v, "name": self.G.nodes[v].get("name", v),
                                   "label": vl}]}
        return {**base, "kind": "unknown", "table": "?", "nodes": []}

    # ------------------------------------------------------------- parsing
    _CIT_RE = re.compile(r"(?i)\b(E\d+|[NDX]-[0-9A-Za-z.\-]+)\b")

    @classmethod
    def _normalize_citations(cls, raw, unit_ids: list[str]) -> tuple[list[str], list[str]]:
        """Map whatever the model put in "citations" onto evidence unit ids.

        Accepts "E123", "[E123]", "e123", 123, "E123, E124"; returns
        (valid ids in evidence order of first mention, invalid tokens)."""
        if raw is None:
            return [], []
        if not isinstance(raw, list):
            raw = [raw]
        lookup = {u.lower(): u for u in unit_ids}
        valid: list[str] = []
        invalid: list[str] = []
        for c in raw:
            if isinstance(c, bool) or c is None:
                invalid.append(str(c))
                continue
            if isinstance(c, (int, float)):
                tokens = [f"E{int(c)}"]
            else:
                s = str(c).strip()
                if not s:
                    continue
                tokens = cls._CIT_RE.findall(s)
                if not tokens:
                    tokens = [f"E{s.strip('[]')}"] if s.strip("[]").isdigit() else [s]
            for tok in tokens:
                tok = tok.rstrip(".-") or tok      # "N-41-9041.00." at a sentence end
                hit = lookup.get(tok.lower())
                if hit is None:
                    if tok not in invalid:
                        invalid.append(tok)
                elif hit not in valid:
                    valid.append(hit)
        return valid, invalid

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        """First {...} object in the reply. strict=False tolerates literal
        newlines/tabs inside strings, which some models emit in long markdown
        answers. A reply cut off by the output-token limit is repaired."""
        raw = raw or ""
        start = raw.find("{")
        if start < 0:
            return None
        end = raw.rfind("}")
        if end > start:
            try:
                obj = json.loads(raw[start:end + 1], strict=False)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Reply was cut off mid-generation (output-token limit): repair by
        # trimming to a syntactically safe point and closing open strings
        # and brackets.
        obj = ChatEngine._repair_truncated_json(raw[start:])
        if isinstance(obj, dict):
            obj["_truncated"] = True
            return obj
        return None

    @staticmethod
    def _repair_truncated_json(s: str) -> dict | None:
        """Try progressively earlier cut points; at each, close any open
        string, then any open brackets, and parse. The first cut that yields
        a dict wins, so as much of the partial answer as possible survives."""
        s = s.rstrip()
        cut_points = [len(s) - 1] + [i for i, ch in enumerate(s)
                                     if ch in ',]}"'][::-1][:400]
        for cut in dict.fromkeys(cut_points):
            cand = s[:cut + 1].rstrip()
            stack, in_str, esc = [], False, False
            for ch in cand:
                if esc:
                    esc = False
                elif in_str:
                    if ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack:
                        stack.pop()
            if in_str:
                if stack and stack[-1] == "]":
                    # a partial array element ("E2" of "E23", half a name) is
                    # wrong data: drop it by trying an earlier cut instead
                    continue
                if cand.endswith("\\"):      # cut inside an escape sequence
                    cand = cand[:-1]
                cand += '"'                  # partial prose answer: keep it
            else:
                cand = cand.rstrip(",")
            closers = "".join(reversed(stack))
            for attempt in (cand + closers,
                            cand.rstrip(":") + closers if cand.endswith(":") else None):
                if attempt is None:
                    continue
                try:
                    obj = json.loads(attempt, strict=False)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _unit_texts(context_text: str) -> dict[str, str]:
        """Map '[unit_id] text' context lines back to {unit_id: full line}."""
        out: dict[str, str] = {}
        for line in context_text.splitlines():
            if line.startswith("[") and "]" in line:
                out[line[1:line.index("]")]] = line
        return out


def _jsonable(d: dict) -> dict:
    """Graph attributes may hold numpy scalars/lists; make them JSON-safe."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            v = v.tolist()
        elif hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and v != v:
            continue
        out[k] = v
    return out
