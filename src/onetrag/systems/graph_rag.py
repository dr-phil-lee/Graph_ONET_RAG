"""S2 — graph RAG: entity linking -> typed subgraph expansion -> serialization.

Edge-type priorities are conditioned on the seed labels: two Job seeds
prioritize converging activity paths (the bridge case); a Job+element pair
prioritizes RATED and ENABLES structure; single-entity questions expand the
local neighborhood by RATED weight.
"""
from __future__ import annotations

import networkx as nx

from ..graph import query as q
from ..graph.serialize import LevelAnchors, edge_line
from ..linking import EntityLinker
from ..llm.client import LLM
from .base import RAGSystem

ACTIVITY_TYPES = {"HAS_TASK", "MAPS_TO", "PART_OF"}
DESCRIPTOR_TYPES = {"RATED", "ENABLES", "IN_GROUP"}


class GraphRAG(RAGSystem):
    name = "s2_graph"

    def __init__(self, cfg: dict, llm: LLM, G: nx.MultiDiGraph,
                 anchors: LevelAnchors, linker: EntityLinker | None = None,
                 oracle_linking: bool = False):
        super().__init__(cfg, llm)
        self.G = G
        self.anchors = anchors
        self.linker = linker or EntityLinker(G)
        self.oracle = oracle_linking
        if oracle_linking:
            self.name = "s2_graph_oracle"

    def seeds_for(self, item: dict) -> list[str]:
        if self.oracle:
            return [e for e in item.get("entities_in_question", []) if e in self.G]
        question = item.get("question_natural") or item["question_template"]
        return self.linker.link(question)

    def retrieve(self, item: dict) -> list[dict]:
        seeds = self.seeds_for(item)
        if not seeds:
            return []
        exp = self.cfg["graph"]["expansion"]
        labels = [self.G.nodes[s].get("label") for s in seeds]
        n_jobs = sum(1 for l in labels if l == "Job")

        question = item.get("question_natural") or item["question_template"]

        # priority prefix, in order: (1) edges directly connecting two seeds,
        # (2) full rollup chains of any seed Task (its MAPS_TO -> PART_OF path
        # is usually the question's subject), (3) RATED edges of a descriptor
        # domain the question names explicitly
        direct = self._direct_edges(seeds)
        ql = question.lower()
        for s, l in zip(seeds, labels):
            if l == "Task":
                direct += self._task_chain(s)
            elif l in ("DWA", "IWA", "GWA"):
                direct += self._activity_neighborhood(s)
            elif l == "Job" and len(seeds) == 1:
                # generic intent heuristics only when the job is the sole seed;
                # with more specific seeds (a Task, a DWA, a second Job) those
                # entities' own structure takes the budget instead
                if "task" in ql or "how many" in ql or "count" in ql:
                    direct += q.tasks_of(self.G, s)
                if "related" in ql:
                    direct += q.out_edges(self.G, s, "RELATED_TO")
                if "how many" in ql or "map to" in ql or "count" in ql:
                    for _, task, _, _ in q.tasks_of(self.G, s):
                        direct += q.out_edges(self.G, task, "MAPS_TO")
        domain = self._domain_in_question(question)
        if domain:
            for s, l in zip(seeds, labels):
                if l == "Job":
                    direct += self._domain_rated(s, domain)

        if n_jobs >= 2:
            edges = self._bridge_edges([s for s, l in zip(seeds, labels) if l == "Job"],
                                       int(exp["max_edges"]))
        elif n_jobs == 1 and len(seeds) > 1:
            edges = q.khop_edges(self.G, seeds, depth=int(exp["max_depth"]),
                                 edge_types=ACTIVITY_TYPES | DESCRIPTOR_TYPES,
                                 max_edges=int(exp["max_edges"]))
        else:
            edges = q.khop_edges(self.G, seeds, depth=2,
                                 edge_types=ACTIVITY_TYPES | DESCRIPTOR_TYPES |
                                 {"RELATED_TO", "USES_TECH"},
                                 max_edges=int(exp["max_edges"]))
        ordered, seen = [], set()
        for e in direct + edges:
            if e[2] not in seen:
                seen.add(e[2])
                ordered.append(e)
        units = [self._node_unit(s) for s in seeds]
        units += [{"unit_id": f"E{k}", "text": edge_line(self.G, (u, v, k, d), self.anchors),
                   "edge_ids": [int(k)], "node_ids": [u, v]} for u, v, k, d in ordered]
        return units

    def _node_unit(self, node: str) -> dict:
        """Seed-node profile line — node properties (description, job zone,
        technology category) are facts no edge carries."""
        a = self.G.nodes[node]
        text = f"{a.get('label')}: {a.get('name', node)}"
        if a.get("label") == "Job":
            jz = a.get("job_zone")
            text = (f"Occupation: {a['name']} (O*NET-SOC {node}). "
                    f"Job Zone: {jz:.0f}. " if jz == jz and jz is not None
                    else f"Occupation: {a['name']} (O*NET-SOC {node}). ")
            text += str(a.get("description", ""))
        elif a.get("description"):
            text += f" — {a['description']}"
        elif a.get("tech_category"):
            text += f" (software category: {a['tech_category']})"
        return {"unit_id": f"N-{node}", "text": text, "edge_ids": [], "node_ids": [node]}

    def _task_chain(self, task: str) -> list[q.Edge]:
        """Task -> DWA -> IWA -> GWA rollup edges (plus the owning HAS_TASK)."""
        out = list(q.in_edges(self.G, task, "HAS_TASK"))
        frontier = [task]
        for etype in ("MAPS_TO", "PART_OF", "PART_OF"):
            nxt = []
            for node in frontier:
                for e in q.out_edges(self.G, node, etype):
                    out.append(e)
                    nxt.append(e[1])
            frontier = nxt
        return out

    def _activity_neighborhood(self, node: str, cap: int = 60) -> list[q.Edge]:
        """For a DWA/IWA/GWA seed: which tasks map into it (and whose they are),
        plus its own hierarchy links."""
        out = list(q.out_edges(self.G, node, "PART_OF"))
        out += q.in_edges(self.G, node, "PART_OF")[:10]
        tasks = q.in_edges(self.G, node, "MAPS_TO")[:cap]
        out += tasks
        for task, _, _, _ in tasks:
            out += q.in_edges(self.G, task, "HAS_TASK")
        return out

    _DOMAIN_KEYWORDS = [("transferable skill", "TransferableSkill"),
                        ("essential skill", "EssentialSkill"),
                        ("knowledge", "Knowledge"), ("abilit", "Ability")]

    def _domain_in_question(self, question: str) -> str | None:
        ql = question.lower()
        for kw, label in self._DOMAIN_KEYWORDS:
            if kw in ql:
                return label
        return None

    def _domain_rated(self, job: str, domain: str, k: int = 25) -> list[q.Edge]:
        edges = [(job, v, key, d) for _, v, key, d in q.out_edges(self.G, job, "RATED")
                 if self.G.nodes[v].get("label") == domain]
        edges.sort(key=lambda e: -(e[3].get("w") or 0.0))
        return edges[:k]

    def _direct_edges(self, seeds: list[str]) -> list[q.Edge]:
        out = []
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                for u, v in ((a, b), (b, a)):
                    data = self.G.get_edge_data(u, v) or {}
                    for k, d in data.items():
                        out.append((u, v, k, d))
        return sorted(out, key=lambda e: e[2])

    def _bridge_edges(self, jobs: list[str], max_edges: int) -> list[q.Edge]:
        """For multi-job questions: put shared-activity converging paths first,
        then each job's own top structure."""
        shared_iwas = sorted(q.shared_activities(self.G, jobs[0], jobs[1], "IWA"))[:12]
        edges: list[q.Edge] = []
        seen: set[int] = set()

        def add(es):
            for e in es:
                if e[2] not in seen and len(edges) < max_edges:
                    seen.add(e[2])
                    edges.append(e)

        for iwa in sorted(shared_iwas):
            for job in jobs[:2]:
                # one representative Task->DWA->IWA path per (job, shared IWA)
                for _, task, tk, td in q.tasks_of(self.G, job):
                    hit = False
                    for _, dwa, dk, dd in q.out_edges(self.G, task, "MAPS_TO"):
                        for _, iw, pk, pd in q.out_edges(self.G, dwa, "PART_OF"):
                            if iw == iwa:
                                add([(job, task, tk, td), (task, dwa, dk, dd),
                                     (dwa, iw, pk, pd)])
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        break
            if len(edges) >= max_edges:
                break
        add(q.khop_edges(self.G, jobs, depth=1, edge_types={"RATED", "RELATED_TO"},
                         max_edges=max_edges - len(edges)))
        return edges
