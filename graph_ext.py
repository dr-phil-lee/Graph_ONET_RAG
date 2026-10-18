"""Chat-graph extension: O*NET occupations outside the research analytic sample.

The research graph (data/graph/) keeps only the 894 occupations that have
importance AND level ratings in all four rated descriptor domains (skills,
abilities, knowledge) plus tasks and task-to-DWA mappings - the right sample
for the benchmark, but it leaves out 29 real O*NET 30.3 occupations whose
descriptor ratings are not yet published, e.g. Data Scientists, Project
Management Specialists, Financial and Investment Analysts, Paramedics,
Penetration Testers. A chatbot user expects to find them.

This module adds those occupations with everything O*NET does publish for
them: title, description, Job Zone, alternate titles, tasks (+ task
importance), task -> DWA mappings, Related Occupations (both directions) and
technology skills. It has no ratings to add, so answers about their skills or
abilities say the evidence does not cover them.

How it is built (deterministic, `python graph_ext.py`):
  1. Run the research ETL (onetrag.graph.build.build_graph, unchanged) with
     the analytic sample widened to every occupation that has tasks and
     task-to-DWA mappings.
  2. Keep only what the research graph lacks: new nodes, and edges whose
     (src, dst, type) is not already in data/graph/.
  3. Number the new edges after the research graph's last edge id, so every
     existing citation id (E<n>) keeps its meaning.
The result is stored in data/graph_ext/ and merged at engine start-up; the
research graph files are never modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

import bootstrap
from onetrag.config import data_path, load_config
from onetrag.graph import store
from onetrag.graph.build import build_graph
from onetrag.onet_io import SOC, OnetData

EXT_DIR = bootstrap.DATA_DIR / "graph_ext"


class _WideOnetData(OnetData):
    """OnetData whose analytic sample also includes occupations that have
    tasks and task-to-DWA mappings but no descriptor ratings."""

    def analytic_sample(self) -> list[str]:
        base = set(super().analytic_sample())
        ts = set(self.load("task_statements.csv")[SOC])
        td = set(self.load("tasks_to_dwas.csv")[SOC])
        occ = set(self.load("occupation_data.csv")[SOC])
        return sorted(base | (ts & td & occ))


def build(onet_dir: str | Path | None = None, out_dir: Path = EXT_DIR) -> dict:
    cfg = load_config()
    data_dir = Path(onet_dir or cfg["onet"]["data_dir"])
    base_dir = data_path(cfg, "graph", "meta.json").parent
    b_nodes, b_edges, b_meta = store.load(base_dir)

    w_nodes, w_edges, _ = build_graph(_WideOnetData(data_dir), cfg)

    have_nodes = set(b_nodes["node_id"])
    have_edges = set(zip(b_edges["src"], b_edges["dst"], b_edges["type"]))
    new_nodes = w_nodes[~w_nodes["node_id"].isin(have_nodes)].reset_index(drop=True)
    keep = [(s, d, t) not in have_edges
            for s, d, t in zip(w_edges["src"], w_edges["dst"], w_edges["type"])]
    new_edges = w_edges[keep].drop(columns=["edge_id"]).reset_index(drop=True)
    start = int(b_edges["edge_id"].max()) + 1
    new_edges.insert(0, "edge_id", (new_edges.index + start).astype("int64"))

    jobs = new_nodes[new_nodes["label"] == "Job"]
    meta = {
        "base_onet_release": b_meta.get("onet_release"),
        "base_analytic_sample": b_meta.get("analytic_sample"),
        "jobs_added": int(len(jobs)),
        "job_ids": sorted(jobs["node_id"]),
        "nodes_added": int(len(new_nodes)),
        "edges_added": int(len(new_edges)),
        "first_edge_id": start,
        "nodes_by_label": new_nodes["label"].value_counts().to_dict(),
        "edges_by_type": new_edges["type"].value_counts().to_dict(),
        "note": ("O*NET 30.3 occupations outside the research analytic sample "
                 "(no published skill/ability/knowledge/work-activity ratings)."),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    new_nodes.to_parquet(out_dir / "nodes.parquet", index=False)
    new_edges.to_parquet(out_dir / "edges.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_meta(ext_dir: Path = EXT_DIR) -> dict | None:
    p = ext_dir / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def merge_into(G: nx.MultiDiGraph, ext_dir: Path = EXT_DIR) -> dict | None:
    """Add the stored extension to G in place; returns its meta (None if absent)."""
    meta = load_meta(ext_dir)
    if meta is None:
        return None
    nodes = pd.read_parquet(ext_dir / "nodes.parquet")
    edges = pd.read_parquet(ext_dir / "edges.parquet")
    ext = store.to_networkx(nodes, edges)
    for n, a in ext.nodes(data=True):
        if n in G:
            continue
        G.add_node(n, **a)
    for u, v, k, a in ext.edges(keys=True, data=True):
        if u not in G or v not in G:
            raise ValueError(f"graph_ext edge {k} references a missing node")
        G.add_edge(u, v, key=k, **a)
    return meta


def merged_stats(base: dict, ext: dict | None) -> dict:
    """data/graph/meta.json counts with the extension added."""
    out = dict(base)
    if not ext:
        return out
    out["analytic_sample_research"] = base.get("analytic_sample")
    out["analytic_sample"] = base.get("analytic_sample", 0) + ext["jobs_added"]
    out["nodes_total"] = base.get("nodes_total", 0) + ext["nodes_added"]
    out["edges_total"] = base.get("edges_total", 0) + ext["edges_added"]
    for key in ("nodes_by_label", "edges_by_type"):
        merged = dict(base.get(key, {}))
        for k, v in ext[key].items():
            merged[k] = merged.get(k, 0) + v
        out[key] = dict(sorted(merged.items(), key=lambda kv: -kv[1]))
    out["extension"] = {k: ext[k] for k in ("jobs_added", "nodes_added", "edges_added",
                                             "first_edge_id", "note")}
    return out


if __name__ == "__main__":
    m = build(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({k: v for k, v in m.items() if k != "job_ids"}, indent=2))
