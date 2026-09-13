"""Parquet <-> NetworkX round-trip for the O*NET graph, plus lookup indexes."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def save(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, meta: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(out / "nodes.parquet", index=False)
    edges_df.to_parquet(out / "edges.parquet", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = Path(out_dir)
    nodes_df = pd.read_parquet(out / "nodes.parquet")
    edges_df = pd.read_parquet(out / "edges.parquet")
    with open(out / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return nodes_df, edges_df, meta


def to_networkx(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.MultiDiGraph:
    """MultiDiGraph with edge key = edge_id; NaN-valued attributes omitted."""
    G = nx.MultiDiGraph()
    for row in nodes_df.to_dict(orient="records"):
        nid = row.pop("node_id")
        G.add_node(nid, **{k: _plain(v) for k, v in row.items() if _keep(v)})
    for row in edges_df.to_dict(orient="records"):
        src, dst, eid = row.pop("src"), row.pop("dst"), row.pop("edge_id")
        G.add_edge(src, dst, key=int(eid), **{k: v for k, v in row.items() if _keep(v)})
    return G


def _keep(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    return True


def _plain(v):
    """Parquet round-trips lists as numpy arrays; convert back for graph attrs."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_graph(out_dir: str | Path) -> nx.MultiDiGraph:
    nodes_df, edges_df, _ = load(out_dir)
    return to_networkx(nodes_df, edges_df)


def name_index(G: nx.MultiDiGraph) -> dict[str, list[str]]:
    """lowercased surface form -> node ids (primary names + Job alt titles)."""
    idx: dict[str, list[str]] = {}

    def add(surface: str, nid: str) -> None:
        idx.setdefault(surface.strip().lower(), []).append(nid)

    for nid, attrs in G.nodes(data=True):
        if attrs.get("name"):
            add(str(attrs["name"]), nid)
        for t in attrs.get("alt_titles") or []:
            add(str(t), nid)
    return idx
