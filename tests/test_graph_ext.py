"""graph_ext: O*NET occupations outside the research sample, and balanced
evidence for comparison questions."""
from __future__ import annotations

import pandas as pd

import bootstrap
import graph_ext

DS = "15-2051.00"      # Data Scientists (no descriptor ratings in O*NET 30.3)
SD = "15-1252.00"      # Software Developers


def test_extension_adds_rating_less_occupations(engine):
    meta = graph_ext.load_meta()
    assert meta["jobs_added"] == 29 and DS in meta["job_ids"]
    for soc in ("13-1082.00", "13-2051.00", "29-2043.00", "15-1299.04"):
        assert soc in meta["job_ids"] and engine.G.nodes[soc]["label"] == "Job"
    a = engine.G.nodes[DS]
    assert a["name"] == "Data Scientists" and a["job_zone"] == 4
    assert engine._title_to_id["Data Scientists"] == DS


def test_extension_keeps_research_edge_ids(engine):
    base = pd.read_parquet(bootstrap.DATA_DIR / "graph" / "edges.parquet")
    ext = pd.read_parquet(graph_ext.EXT_DIR / "edges.parquet")
    assert ext["edge_id"].min() == base["edge_id"].max() + 1
    assert not set(ext["edge_id"]) & set(base["edge_id"])
    keys = set(zip(base["src"], base["dst"], base["type"]))
    assert not any(k in keys for k in zip(ext["src"], ext["dst"], ext["type"]))
    # research edge 0 is unchanged
    u, v, k, d = engine.edge_index[0]
    assert (u, v, d["type"]) == tuple(base.loc[0, ["src", "dst", "type"]])


def test_extension_job_has_tasks_related_and_tech_but_no_ratings(engine):
    types = {d["type"] for _, _, d in engine.G.out_edges(DS, data=True)}
    assert {"HAS_TASK", "RELATED_TO", "USES_TECH"} <= types and "RATED" not in types
    # related occupations point back into the research graph and vice versa
    assert any(u == SD for u, _, d in engine.G.in_edges(DS, data=True)
               if d["type"] == "RELATED_TO")


def test_linker_finds_data_scientists(engine):
    assert DS in engine.linker.link("Compare Software Developers and Data Scientists.")


def test_comparison_evidence_covers_both_occupations(engine, fake_llm):
    ans = engine.ask("Compare Software Developers and Data Scientists.")
    assert set(ans.seeds) == {SD, DS}
    ids = [u for u, _, _ in ans.evidence]
    assert f"N-{SD}" in ids and f"N-{DS}" in ids
    text = {u: t for u, t, _ in ans.evidence}
    ds_tasks = [u for u, t in text.items() if t.startswith(f"[{u}] Data Scientists has task")]
    sd_tasks = [u for u, t in text.items()
                if t.startswith(f"[{u}] Software Developers has task")]
    assert len(ds_tasks) >= 5 and len(sd_tasks) >= 5
    assert "no skill, ability, knowledge or work-activity ratings" in text[f"N-{DS}"]
    assert "no skill" not in text[f"N-{SD}"]
    # both compared occupations get a card, in the order asked
    assert [c["soc"] for c in ans.occupations][:2] == [SD, DS]
    assert ans.occupations[0]["ai_exposure"]["composite"] == 0.8254
    assert ans.occupations[1]["ai_exposure"] is None


def test_related_ranks_are_whole_numbers(engine):
    """Evidence lines must read "rank 2", not the parquet float "rank 2.0"."""
    ranks = [d["index"] for _, _, d in engine.G.edges(data=True) if d["type"] == "RELATED_TO"]
    assert ranks and all(isinstance(r, int) for r in ranks)
    e = next(k for k, (u, v, _, d) in engine.edge_index.items()
             if d["type"] == "RELATED_TO" and u == "41-9041.00" and d["index"] == 2)
    assert "rank 2)" in engine.rag._edge_unit(engine.edge_index[e])["text"]


def test_every_index_row_joins_to_its_own_occupation(engine):
    import ai_exposure
    idx = ai_exposure.default_index()
    joined = {x["soc"] for _, s in engine.occupations if (x := engine.exposure_for(s))}
    assert joined == set(idx.rows)            # all 696 rows, each exactly once
    for _, s in engine.occupations:
        x = engine.exposure_for(s)
        if x:
            assert x["index_title"].lower() == engine.G.nodes[s]["name"].lower()
