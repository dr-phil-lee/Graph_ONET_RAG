"""ChatEngine: one-time loading of graph + linker + LLM, per-question ask().

Wires the frozen research S2 stack (imported, never modified) to an
OpenRouter-hosted generator model. Config overrides happen in memory only —
Research/config.yaml is never touched.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
from onetrag.config import data_path, load_config
from onetrag.graph import store
from onetrag.graph.serialize import LevelAnchors
from onetrag.linking import EntityLinker
from onetrag.onet_io import OnetData

from hybrid_graph_rag import HybridGraphRAG
from openrouter_llm import OpenRouterLLM


class _MissingKeyLLM:
    """Placeholder until the user supplies an OpenRouter API key."""

    description = "not configured — enter an OpenRouter API key"

    def complete(self, **kwargs) -> dict:
        raise RuntimeError("No OpenRouter API key is set. Enter your key in "
                           "the sidebar and click 'Apply model'.")

    def count_tokens(self, text: str, model: str | None = None) -> tuple[int, str]:
        return max(1, len(text) // 4), "estimate"


@dataclass
class Answer:
    question: str
    parsed: dict | None                       # {"answer","citations","reasoning"} or None
    raw: str
    seeds: list[str]                          # node ids whose profiles entered the context
    evidence: list[tuple[str, str, bool]]     # (unit_id, text, cited)
    cited_edge_ids: list[int]
    retrieved_edge_ids: list[int]
    usage: dict
    context_tokens: int
    rec: dict = field(repr=False, default_factory=dict)


class ChatEngine:
    def __init__(self, context_budget: int = bootstrap.DEFAULT_CONTEXT_BUDGET):
        cfg = load_config()
        cfg["budget"]["context_tokens"] = int(context_budget)     # in memory only
        cfg["models"]["generator"] = "openrouter"                 # label only
        self.cfg = cfg
        self.G = store.load_graph(data_path(cfg, "graph", "meta.json").parent)
        self.anchors = self._load_anchors(cfg)
        self.linker = EntityLinker(self.G)
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        llm = (OpenRouterLLM(key, bootstrap.OPENROUTER_DEFAULT_MODEL)
               if key else _MissingKeyLLM())
        self.rag = HybridGraphRAG(cfg, llm, self.G, self.anchors, self.linker)
        self.edge_index = {k: (u, v, k, d)
                           for u, v, k, d in self.G.edges(keys=True, data=True)}
        self.occupations = sorted(
            (self.G.nodes[n]["name"], n) for n in self.G
            if self.G.nodes[n].get("label") == "Job")
        self._n_asked = 0

    @staticmethod
    def _load_anchors(cfg: dict) -> LevelAnchors | None:
        """Behavioral level anchors: the bundled data/onet/ file first, then a
        full O*NET CSV folder if configured; None just drops the anchor
        phrases from RATED evidence lines."""
        bundled = bootstrap.DATA_DIR / "onet"
        for candidate in (bundled, cfg["onet"]["data_dir"]):
            try:
                return LevelAnchors(OnetData(candidate))
            except (FileNotFoundError, OSError):
                continue
        return None

    # ------------------------------------------------------------- backend
    @property
    def backend_description(self) -> str:
        return getattr(self.rag.llm, "description", type(self.rag.llm).__name__)

    @property
    def is_configured(self) -> bool:
        return isinstance(self.rag.llm, OpenRouterLLM)

    def set_backend(self, api_key: str, model: str | None = None) -> str:
        """Swap the OpenRouter model/key; graph and linker stay loaded."""
        self.rag.llm = OpenRouterLLM(
            api_key, model or bootstrap.OPENROUTER_DEFAULT_MODEL)
        self.cfg["models"]["generator"] = self.rag.llm.model
        return self.backend_description

    def set_budget(self, tokens: int) -> None:
        self.rag.budget = int(tokens)

    # ------------------------------------------------------------- questions
    def preview_seeds(self, question: str) -> list[tuple[str, str, str]]:
        """(node_id, name, label) the linker would use — shown live in the UI."""
        out = []
        for n in self.linker.link(question):
            a = self.G.nodes[n]
            out.append((n, a.get("name", n), a.get("label", "?")))
        return out

    def node_info(self, node_id: str) -> tuple[str, str]:
        a = self.G.nodes[node_id]
        return a.get("name", node_id), a.get("label", "?")

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
        between versus every daily technology technologies""".split())

    @staticmethod
    def _normalize_word(w: str) -> str:
        w = w.removesuffix("'s").rstrip("'")
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # naive singular: teachers -> teacher
        return w

    def _question_ngrams(self, question: str) -> tuple[list[str], list[str]]:
        """(bigrams, unigrams) of normalized content words, in question order.
        Bigrams catch multi-word occupations ("data analyst", "air traffic")."""
        import re

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

    def _match_term(self, term: str) -> list[tuple[str, str]]:
        """Occupations whose title (else alternate title) contains `term` as
        whole words, most generic first: [(node_id, title), ...]."""
        import re

        pat = re.compile(rf"\b{re.escape(term)}s?\b")
        hits = [(title, nid) for title, nid in self.occupations
                if pat.search(title.lower())]
        if hits:
            # fewer words, then shorter = more generic title first
            hits.sort(key=lambda t: (len(t[0].split()), len(t[0]), t[0]))
            return [(nid, title) for title, nid in hits]
        scored = []  # alternate-title fallback: best matching alt per job
        for title, nid in self.occupations:
            best = None
            for alt in self.G.nodes[nid].get("alt_titles") or []:
                al = str(alt).lower()
                if pat.search(al):
                    rank = (0 if al in (term, term + "s") else
                            1 if al.startswith(term) else 2,
                            len(al))
                    if best is None or rank < best:
                        best = rank
            if best is not None:
                scored.append((best, title, nid))
        scored.sort()
        return [(nid, title) for _, title, nid in scored]

    def suggest_occupations(self, question: str,
                            per_term: int = 5) -> dict[str, list[tuple[str, str]]]:
        """{question term: [(node_id, title), ...]} for every content word or
        word pair that names an occupation anywhere in the 894 titles or the
        48,752 alternate titles. A matched bigram absorbs its unigrams."""
        bigrams, unigrams = self._question_ngrams(question)
        suggestions: dict[str, list[tuple[str, str]]] = {}
        used_words: set[str] = set()
        for bg in bigrams:
            m = self._match_term(bg)
            if m:
                suggestions[bg] = m[:per_term]
                used_words |= set(bg.split())
        for term in unigrams:
            if term in used_words:
                continue
            m = self._match_term(term)
            if m:
                suggestions[term] = m[:per_term]
        return suggestions

    def auto_occupations(self, question: str, max_jobs: int = 3,
                         exclude: list[str] | tuple = ()) -> list[tuple[str, str]]:
        """Best-guess occupation per question term: [(node_id, title), ...].
        Terms already covered by an `exclude` job's title (a linked or pinned
        occupation) are skipped, so this only fills the gaps the exact-match
        entity linker left."""
        import re

        excluded_titles = " | ".join(
            self.G.nodes[n].get("name", "").lower()
            for n in exclude if n in self.G)
        out: list[tuple[str, str]] = []
        seen: set[str] = set(exclude)
        for term, cands in self.suggest_occupations(question).items():
            if excluded_titles and re.search(rf"\b{re.escape(term)}s?\b",
                                             excluded_titles):
                continue
            nid, title = cands[0]
            if nid not in seen:
                seen.add(nid)
                out.append((nid, title))
            if len(out) >= max_jobs:
                break
        return out

    def ask(self, question: str, forced_seeds: list[str] | None = None,
            qid: str | None = None) -> Answer:
        self._n_asked += 1
        item = {"qid": qid or f"chat-{self._n_asked}-{time.time():.0f}",
                "question_template": question, "question_natural": None,
                "forced_seeds": forced_seeds or []}
        rec = self.rag.answer(item)

        parsed = self._parse_json(rec["raw"])
        cited_units: set[str] = set()
        if parsed and isinstance(parsed.get("citations"), list):
            cited_units = {str(c).strip().strip("[]") for c in parsed["citations"]}
        cited_edges = sorted({e for uid in cited_units
                              for e in rec["unit_edge_ids"].get(uid, [])})

        texts = self._unit_texts(rec["context_text"])
        evidence = [(uid, texts.get(uid, ""), uid in cited_units)
                    for uid in rec["context_units"]]
        seeds = [uid[2:] for uid in rec["context_units"] if uid.startswith("N-")]

        return Answer(question=question, parsed=parsed, raw=rec["raw"],
                      seeds=seeds, evidence=evidence,
                      cited_edge_ids=cited_edges,
                      retrieved_edge_ids=list(rec["context_edge_ids"]),
                      usage=rec["usage"], context_tokens=rec["context_tokens"],
                      rec=rec)

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        start = raw.find("{")
        if start < 0:
            return None
        end = raw.rfind("}")
        if end > start:
            try:
                obj = json.loads(raw[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Reply was cut off mid-generation (output-token limit): repair by
        # trimming to a syntactically safe point and closing open brackets.
        obj = ChatEngine._repair_truncated_json(raw[start:])
        if isinstance(obj, dict):
            obj["_truncated"] = True
            return obj
        return None

    @staticmethod
    def _repair_truncated_json(s: str) -> dict | None:
        cut_points = [i for i, ch in enumerate(s)
                      if ch in ',]}"'][::-1][:400]
        for cut in cut_points:
            cand = s[:cut + 1].rstrip().rstrip(",")
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
                continue
            try:
                obj = json.loads(cand + "".join(reversed(stack)))
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
