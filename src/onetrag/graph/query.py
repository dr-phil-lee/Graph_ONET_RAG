"""Typed traversals over the O*NET MultiDiGraph.

Load-bearing module: these functions generate the benchmark's gold answers
AND drive graph-RAG retrieval, so they are unit-tested hard. Every function
takes the graph as its first argument and returns plain Python data.

Edge convention (see build.py): each edge's key is its stable edge_id, used
for gold-evidence bookkeeping and citation scoring.
"""
from __future__ import annotations

from collections import deque

import networkx as nx

Edge = tuple[str, str, int, dict]  # (src, dst, edge_id, attrs)


def out_edges(G: nx.MultiDiGraph, node: str, etype: str) -> list[Edge]:
    if node not in G:
        return []
    return [(u, v, k, d) for u, v, k, d in G.out_edges(node, keys=True, data=True)
            if d.get("type") == etype]


def in_edges(G: nx.MultiDiGraph, node: str, etype: str) -> list[Edge]:
    if node not in G:
        return []
    return [(u, v, k, d) for u, v, k, d in G.in_edges(node, keys=True, data=True)
            if d.get("type") == etype]


def label(G: nx.MultiDiGraph, node: str) -> str | None:
    return G.nodes[node].get("label") if node in G else None


def name(G: nx.MultiDiGraph, node: str) -> str | None:
    return G.nodes[node].get("name") if node in G else None


# ---------------------------------------------------------------- occupations

def tasks_of(G, job: str, core_only: bool = False) -> list[Edge]:
    es = out_edges(G, job, "HAS_TASK")
    if core_only:
        es = [e for e in es if e[3].get("task_type") == "Core"]
    return es


def dwas_of_task(G, task: str) -> set[str]:
    return {v for _, v, _, _ in out_edges(G, task, "MAPS_TO")}


def dwas_of_job(G, job: str) -> set[str]:
    out: set[str] = set()
    for _, task, _, _ in tasks_of(G, job):
        out |= dwas_of_task(G, task)
    return out


def parents_of(G, node: str) -> set[str]:
    """PART_OF parents (DWA -> IWA, IWA -> GWA)."""
    return {v for _, v, _, _ in out_edges(G, node, "PART_OF")}


def iwas_of_job(G, job: str) -> set[str]:
    out: set[str] = set()
    for dwa in dwas_of_job(G, job):
        out |= parents_of(G, dwa)
    return out


def gwas_of_job_via_tasks(G, job: str) -> set[str]:
    out: set[str] = set()
    for iwa in iwas_of_job(G, job):
        out |= parents_of(G, iwa)
    return out


def shared_activities(G, job_a: str, job_b: str, level: str = "IWA") -> set[str]:
    fn = {"DWA": dwas_of_job, "IWA": iwas_of_job, "GWA": gwas_of_job_via_tasks}[level]
    return fn(G, job_a) & fn(G, job_b)


def jobs_with_dwa(G, dwa: str) -> set[str]:
    jobs: set[str] = set()
    for task, _, _, _ in in_edges(G, dwa, "MAPS_TO"):
        for j, _, _, _ in in_edges(G, task, "HAS_TASK"):
            jobs.add(j)
    return jobs


def related_jobs(G, job: str, tiers: set[str] | None = None) -> list[tuple[str, str, int]]:
    """[(related job, tier, index)] sorted by O*NET's own relatedness index."""
    out = [(v, d.get("tier"), d.get("index", 0))
           for _, v, _, d in out_edges(G, job, "RELATED_TO")
           if tiers is None or d.get("tier") in tiers]
    return sorted(out, key=lambda t: t[2])


# ------------------------------------------------------------------- ratings

def rating(G, job: str, element: str) -> dict | None:
    for _, v, _, d in out_edges(G, job, "RATED"):
        if v == element:
            return d
    return None


def rated_elements(G, job: str, elem_label: str,
                   include_not_relevant: bool = False) -> list[tuple[str, dict]]:
    out = []
    for _, v, _, d in out_edges(G, job, "RATED"):
        if label(G, v) == elem_label and (include_not_relevant or not d.get("not_relevant")):
            out.append((v, d))
    return out


def top_rated(G, job: str, elem_label: str, k: int = 10,
              key: str = "importance") -> list[tuple[str, dict]]:
    """Top-k elements of a label for a job, descending by importance (or level).

    Ties broken by the other scale then element id, so results are deterministic.
    """
    other = "level" if key == "importance" else "importance"
    rated = rated_elements(G, job, elem_label)
    rated.sort(key=lambda t: (-(t[1].get(key) or 0), -(t[1].get(other) or 0), t[0]))
    return rated[:k]


def not_relevant_elements(G, job: str, elem_label: str) -> list[str]:
    return [v for _, v, _, d in out_edges(G, job, "RATED")
            if label(G, v) == elem_label and d.get("not_relevant")]


# ----------------------------------------------------------- structural paths

def enabled_gwas(G, element: str) -> set[str]:
    return {v for _, v, _, _ in out_edges(G, element, "ENABLES")}


def evidence_paths_task_to_element(G, task: str, element: str) -> list[list[str]]:
    """All Task -> DWA -> IWA -> GWA <- ENABLES <- element node paths."""
    targets = enabled_gwas(G, element)
    paths = []
    for dwa in dwas_of_task(G, task):
        for iwa in parents_of(G, dwa):
            for gwa in parents_of(G, iwa):
                if gwa in targets:
                    paths.append([task, dwa, iwa, gwa, element])
    return paths


def _edge_weight(d: dict) -> float:
    """Expansion priority: RATED edges by w, HAS_TASK by task importance/5,
    structural edges (MAPS_TO/PART_OF/ENABLES) medium, the rest low."""
    if d.get("w") is not None:
        return float(d["w"])
    if d.get("type") == "HAS_TASK":
        imp = d.get("importance")
        return float(imp) / 5.0 if imp is not None else 0.5
    if d.get("type") in ("MAPS_TO", "PART_OF", "ENABLES"):
        return 0.6
    return 0.3


def khop_edges(G, seeds: list[str], depth: int = 2,
               edge_types: set[str] | None = None,
               max_edges: int = 120) -> list[Edge]:
    """Breadth-first typed expansion from seed nodes, traversing both directions.

    Returns edges in BFS discovery order (so truncation keeps the closest
    structure). RATED edges are ordered by weight w within each frontier.
    """
    seen_nodes = set(s for s in seeds if s in G)
    seen_edges: set[int] = set()
    result: list[Edge] = []
    frontier = deque(seen_nodes)
    for _ in range(depth):
        if not frontier or len(result) >= max_edges:
            break
        next_frontier: deque[str] = deque()
        level_edges: list[Edge] = []
        for node in frontier:
            for u, v, k, d in list(G.out_edges(node, keys=True, data=True)) + \
                              list(G.in_edges(node, keys=True, data=True)):
                if edge_types is not None and d.get("type") not in edge_types:
                    continue
                if k in seen_edges:
                    continue
                seen_edges.add(k)
                level_edges.append((u, v, k, d))
        level_edges.sort(key=lambda e: (-_edge_weight(e[3]), e[2]))
        for u, v, k, d in level_edges:
            if len(result) >= max_edges:
                break
            result.append((u, v, k, d))
            for n in (u, v):
                if n not in seen_nodes:
                    seen_nodes.add(n)
                    next_frontier.append(n)
        frontier = next_frontier
    return result
