"""Subgraph -> numbered evidence lines for LLM prompts.

Each line carries its edge_id tag ("[E123]") so generated citations can be
scored deterministically against gold evidence. Level values are verbalized
with the official O*NET behavioral anchors (level_scale_anchors.csv) — the
same verbalization the vector-RAG descriptor cards use, for fairness.
"""
from __future__ import annotations

import networkx as nx

from ..onet_io import OnetData
from .query import Edge


class LevelAnchors:
    """element_id -> [(anchor value, description)]; nearest-anchor verbalizer."""

    def __init__(self, data: OnetData):
        df = data.load("level_scale_anchors.csv")
        df = df[df["Scale ID"] == "LV"]
        self._anchors: dict[str, list[tuple[float, str]]] = {}
        for eid, grp in df.groupby("Element ID"):
            self._anchors[eid] = sorted(
                zip(grp["Anchor Value"].astype(float), grp["Anchor Description"]))

    def verbalize(self, element_id: str, level: float) -> str | None:
        anchors = self._anchors.get(element_id)
        if not anchors:
            return None
        below = [a for a in anchors if a[0] <= level]
        above = [a for a in anchors if a[0] >= level]
        lo = below[-1] if below else None
        hi = above[0] if above else None
        if lo and hi and lo[0] == hi[0]:
            return f"at the level of '{lo[1]}' ({lo[0]:.0f})"
        parts = []
        if lo:
            parts.append(f"above '{lo[1]}' ({lo[0]:.0f})")
        if hi:
            parts.append(f"below '{hi[1]}' ({hi[0]:.0f})")
        return " and ".join(parts) if parts else None


def edge_line(G: nx.MultiDiGraph, edge: Edge, anchors: LevelAnchors | None = None) -> str:
    u, v, eid, d = edge
    etype = d.get("type")
    un = G.nodes[u].get("name", u) if u in G else u
    vn = G.nodes[v].get("name", v) if v in G else v
    ul = G.nodes[u].get("label", "") if u in G else ""
    vl = G.nodes[v].get("label", "") if v in G else ""

    if etype == "HAS_TASK":
        extra = []
        if d.get("task_type"):
            extra.append(str(d["task_type"]).lower())
        if d.get("importance") is not None:
            extra.append(f"importance {d['importance']:.1f}/5")
        suffix = f" ({', '.join(extra)})" if extra else ""
        return f"[E{eid}] {un} has task{suffix}: \"{vn}\""
    if etype == "MAPS_TO":
        return f"[E{eid}] Task \"{_short(un)}\" involves the detailed work activity: {vn}"
    if etype == "PART_OF":
        return f"[E{eid}] {ul} \"{un}\" is part of the {vl.lower()} \"{vn}\""
    if etype == "ENABLES":
        return f"[E{eid}] The {_article_label(ul)} \"{un}\" enables the general work activity \"{vn}\""
    if etype == "RATED":
        if d.get("not_relevant"):
            return f"[E{eid}] For {un}, the {_article_label(vl)} \"{vn}\" is marked NOT RELEVANT"
        parts = []
        if d.get("importance") is not None:
            parts.append(f"importance {d['importance']:.2f}/5")
        if d.get("level") is not None:
            lvl = f"level {d['level']:.2f}/7"
            if anchors is not None:
                verb = anchors.verbalize(v, float(d["level"]))
                if verb:
                    lvl += f" ({verb})"
            parts.append(lvl)
        return f"[E{eid}] {un} rates the {_article_label(vl)} \"{vn}\": {', '.join(parts)}"
    if etype == "RELATED_TO":
        return f"[E{eid}] {un} is related to the occupation {vn} (tier: {d.get('tier')}, rank {d.get('index')})"
    if etype == "USES_TECH":
        flags = []
        if d.get("hot_technology"):
            flags.append("hot technology")
        if d.get("in_demand"):
            flags.append("in demand")
        suffix = f" ({', '.join(flags)})" if flags else ""
        return f"[E{eid}] {un} uses the software \"{vn}\"{suffix}"
    if etype == "IN_GROUP":
        return f"[E{eid}] \"{un}\" belongs to the element group \"{vn}\""
    return f"[E{eid}] {un} -{etype}-> {vn}"


def serialize_edges(G: nx.MultiDiGraph, edges: list[Edge],
                    anchors: LevelAnchors | None = None) -> str:
    """Evidence block: edges grouped by type in a stable, readable order."""
    order = ["HAS_TASK", "MAPS_TO", "PART_OF", "ENABLES", "RATED",
             "RELATED_TO", "USES_TECH", "IN_GROUP"]
    rank = {t: i for i, t in enumerate(order)}
    ordered = sorted(edges, key=lambda e: (rank.get(e[3].get("type"), 99), e[2]))
    return "\n".join(edge_line(G, e, anchors) for e in ordered)


def _short(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _article_label(label: str) -> str:
    mapping = {"EssentialSkill": "essential skill", "TransferableSkill": "transferable skill",
               "Ability": "ability", "Knowledge": "knowledge area", "WorkStyle": "work style",
               "GWA": "general work activity", "IWA": "intermediate work activity",
               "DWA": "detailed work activity", "Technology": "software"}
    return mapping.get(label, label.lower() if label else "element")
