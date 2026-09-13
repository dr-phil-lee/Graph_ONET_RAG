"""Vector-RAG corpus: one "card" per retrieval unit, built from the graph.

Every card records the node_ids and edge_ids it verbalizes, which makes
vector-RAG citations exactly as checkable against gold evidence as graph
citations. Card text deliberately includes the DWA hierarchy names and the
same level-anchor verbalization the graph serializer uses — the vector
baseline sees the same information, differently packaged (fairness).

Card kinds: job (894), task (~18k), descriptor (one per job x domain, ~4.5k),
hierarchy (one per DWA/IWA/GWA, ~2.5k).
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from ..graph import query as q
from ..graph.serialize import LevelAnchors

DESCRIPTOR_DOMAINS = ["EssentialSkill", "TransferableSkill", "Ability", "Knowledge", "GWA"]
DOMAIN_PHRASE = {
    "EssentialSkill": "essential skills", "TransferableSkill": "transferable skills",
    "Ability": "abilities", "Knowledge": "knowledge areas", "GWA": "general work activities",
}


def build_corpus(G: nx.MultiDiGraph, anchors: LevelAnchors,
                 jobs: list[str] | None = None, top_k_descriptors: int = 10,
                 hierarchy_example_jobs: int = 8) -> list[dict]:
    if jobs is None:
        jobs = sorted(n for n, d in G.nodes(data=True) if d.get("label") == "Job")
    jobset = set(jobs)
    docs: list[dict] = []

    # ---- job profile cards ----
    for job in jobs:
        a = G.nodes[job]
        alt = (a.get("alt_titles") or [])[:8]
        jz = a.get("job_zone")
        jz_txt = f"{jz:.0f}" if isinstance(jz, float) and jz == jz else "unknown"
        text = (f"Occupation: {a['name']} (O*NET-SOC {job}). {a.get('description', '')} "
                f"Job Zone: {jz_txt}.")
        if alt:
            text += f" Also known as: {'; '.join(alt)}."
        docs.append(_card(f"D-job-{job}", "job", text, [job], []))

    # ---- task cards ----
    for job in jobs:
        jname = G.nodes[job]["name"]
        for _, task, eid, d in q.tasks_of(G, job):
            dwa_edges = q.out_edges(G, task, "MAPS_TO")
            dwa_names = [G.nodes[v]["name"] for _, v, _, _ in dwa_edges]
            ttype = f" ({d['task_type']}" if d.get("task_type") else ""
            if d.get("importance") is not None:
                ttype += f", importance {d['importance']:.1f}/5" if ttype else \
                         f" (importance {d['importance']:.1f}/5"
            ttype += ")" if ttype else ""
            text = f"{jname} — task{ttype}: \"{G.nodes[task]['name']}\""
            if dwa_names:
                text += f" This task involves the detailed work activities: {'; '.join(dwa_names)}."
            docs.append(_card(f"D-task-{job}-{task}", "task", text,
                              [job, task] + [v for _, v, _, _ in dwa_edges],
                              [eid] + [k for _, _, k, _ in dwa_edges]))

    # ---- descriptor cards: ALL rated elements per (job, domain), so every
    # RATED fact exists somewhere in the corpus (fairness: exact-figure
    # questions must be answerable by the vector baseline in principle).
    # Level-anchor verbalization on the top entries keeps card size bounded.
    for job in jobs:
        jname = G.nodes[job]["name"]
        edge_of = {v: k for _, v, k, _ in q.out_edges(G, job, "RATED")}
        for domain in DESCRIPTOR_DOMAINS:
            rated = q.top_rated(G, job, domain, k=10_000)  # all, sorted by importance
            if not rated:
                continue
            lines, node_ids, edge_ids = [], [job], []
            for rank, (elem, d) in enumerate(rated):
                node_ids.append(elem)
                edge_ids.append(edge_of[elem])
                parts = []
                if d.get("importance") is not None:
                    parts.append(f"importance {d['importance']:.2f}/5")
                if d.get("level") is not None:
                    lvl = f"level {d['level']:.2f}/7"
                    if rank < top_k_descriptors:
                        verb = anchors.verbalize(elem, float(d["level"]))
                        if verb:
                            lvl += f" ({verb})"
                    parts.append(lvl)
                lines.append(f"{G.nodes[elem]['name']}: {', '.join(parts)}")
            text = (f"{DOMAIN_PHRASE[domain].capitalize()} ratings for {jname}, "
                    f"sorted by importance: " + " | ".join(lines))
            nr = q.not_relevant_elements(G, job, domain)
            if nr:
                text += (" | Marked NOT RELEVANT for this occupation: "
                         + "; ".join(G.nodes[e]["name"] for e in sorted(nr)))
                node_ids += sorted(nr)
                edge_ids += [edge_of[e] for e in sorted(nr)]
            docs.append(_card(f"D-desc-{job}-{domain}", "descriptor", text, node_ids, edge_ids))

    # ---- hierarchy cards ----
    for node, attrs in G.nodes(data=True):
        label = attrs.get("label")
        if label not in ("DWA", "IWA", "GWA"):
            continue
        chain, cur, edge_ids = [], node, []
        while True:
            parents = q.out_edges(G, cur, "PART_OF")
            if not parents:
                break
            _, parent, k, _ = parents[0]
            chain.append(G.nodes[parent]["name"])
            edge_ids.append(k)
            cur = parent
        text = f"{_LONG[label]}: \"{attrs['name']}\"."
        if chain:
            text += f" Part of: {' > '.join(reversed(chain))}."
        if label == "DWA":
            ex = sorted(q.jobs_with_dwa(G, node) & jobset)[:hierarchy_example_jobs]
            if ex:
                names = [G.nodes[j]["name"] for j in ex]
                text += f" Example occupations performing it: {'; '.join(names)}."
        docs.append(_card(f"D-hier-{node}", "hierarchy", text, [node], edge_ids))

    return docs


_LONG = {"DWA": "Detailed work activity", "IWA": "Intermediate work activity",
         "GWA": "General work activity"}


def _card(doc_id: str, kind: str, text: str, node_ids: list[str], edge_ids: list[int]) -> dict:
    return {"doc_id": doc_id, "kind": kind, "text": text,
            "node_ids": node_ids, "edge_ids": [int(e) for e in edge_ids]}


def save_corpus(docs: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
