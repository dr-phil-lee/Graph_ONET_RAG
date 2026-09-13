"""Deterministic ETL: O*NET CSVs -> node/edge DataFrames over the analytic sample.

Schema (extends the sibling project's validated schema for RAG use):

  (:Job)-[:HAS_TASK {task_type, importance}]->(:Task)
  (:Task)-[:MAPS_TO]->(:DWA)-[:PART_OF]->(:IWA)-[:PART_OF]->(:GWA)
  (:EssentialSkill|:TransferableSkill|:Ability|:WorkStyle)-[:ENABLES]->(:GWA)
  (:Job)-[:RATED {importance, level, not_relevant, w}]->(element | :GWA)
  (:Job)-[:RELATED_TO {tier, index}]->(:Job)
  (:Job)-[:USES_TECH {hot_technology, in_demand}]->(:Technology)
  (element)-[:IN_GROUP]->(:ElementGroup)

New vs. the sibling schema: Technology nodes (software_skills.csv), HAS_TASK
importance weights (task_ratings IM), Job.alt_titles (entity linking), element
descriptions (content_model_reference), a per-edge stable edge_id (citation
scoring), and a RATED pruning weight w = a*(IM/5) + b*(LV/7).

Rows with `Recommend Suppress == Y` are dropped and the counts logged in meta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..onet_io import RATED_DOMAINS, SOC, OnetData

TECH_PREFIX = "TECH::"


def build_graph(data: OnetData, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (nodes_df, edges_df, meta). Pure function of the CSVs + config."""
    jobs = data.analytic_sample()
    jobset = set(jobs)
    wa, wb = cfg["graph"]["rated_weight"]["importance"], cfg["graph"]["rated_weight"]["level"]
    suppressed: dict[str, int] = {}
    nodes: list[dict] = []
    edges: list[dict] = []

    # ---- Job nodes ----
    occ = data.load("occupation_data.csv").set_index(SOC)
    jz = data.load("job_zones.csv").set_index(SOC)["Job Zone"]
    alt = data.alt_titles()
    for s in jobs:
        nodes.append({
            "node_id": s, "label": "Job", "name": occ.loc[s, "Title"],
            "description": occ.loc[s, "Description"],
            "job_zone": float(jz.get(s, np.nan)),
            "alt_titles": list(alt.get(s, [])),
        })

    # ---- Task nodes + HAS_TASK (with task-level importance) ----
    ts = data.load("task_statements.csv")
    t = ts[ts[SOC].isin(jobset)].drop_duplicates("Task ID")[["Task ID", "Task", "Task Type"]]
    for tid, text, ttype in t.itertuples(index=False):
        nodes.append({"node_id": str(tid), "label": "Task", "name": text,
                      "task_type": ttype if pd.notna(ttype) else None})
    tim = data.task_importance().set_index([SOC, "Task ID"])["importance"]
    ht = ts[ts[SOC].isin(jobset)][[SOC, "Task ID", "Task Type"]]
    for soc, tid, ttype in ht.itertuples(index=False):
        imp = tim.get((soc, tid), np.nan)
        edges.append({"src": soc, "dst": str(tid), "type": "HAS_TASK",
                      "task_type": ttype if pd.notna(ttype) else None,
                      "importance": float(imp) if pd.notna(imp) else None})

    # ---- Activity hierarchy: DWA / IWA / GWA nodes + PART_OF ----
    h = data.load("gwas_to_iwas_to_dwas.csv")
    for col_id, col_name, label in [("DWA Element ID", "DWA Element Name", "DWA"),
                                    ("IWA Element ID", "IWA Element Name", "IWA"),
                                    ("GWA Element ID", "GWA Element Name", "GWA")]:
        for eid, name in h[[col_id, col_name]].drop_duplicates(col_id).itertuples(index=False):
            nodes.append({"node_id": eid, "label": label, "name": name})
    for d_, i_ in h[["DWA Element ID", "IWA Element ID"]].drop_duplicates().itertuples(index=False):
        edges.append({"src": d_, "dst": i_, "type": "PART_OF"})
    for i_, g_ in h[["IWA Element ID", "GWA Element ID"]].drop_duplicates().itertuples(index=False):
        edges.append({"src": i_, "dst": g_, "type": "PART_OF"})

    # ---- MAPS_TO: Task -> DWA ----
    td = data.load("tasks_to_dwas.csv")
    known_tasks = set(t["Task ID"].astype(str))
    mt = td[td[SOC].isin(jobset)][["Task ID", "DWA Element ID"]].drop_duplicates()
    for tid, dwa in mt.itertuples(index=False):
        if str(tid) in known_tasks:
            edges.append({"src": str(tid), "dst": dwa, "type": "MAPS_TO"})

    # ---- element nodes, RATED, ENABLES, IN_GROUP ----
    cmr = data.load("content_model_reference.csv").set_index("Element ID")
    group_ids: set[str] = set()

    def add_rated(ratings: pd.DataFrame) -> None:
        for r in ratings.itertuples(index=False):
            im, lv = r.importance, r.level
            w = None
            if pd.notna(im) and pd.notna(lv):
                w = round(wa * (im / 5.0) + wb * (lv / 7.0), 4)
            edges.append({
                "src": r[0], "dst": r[1], "type": "RATED",
                "importance": im if pd.notna(im) else None,
                "level": lv if pd.notna(lv) else None,
                "not_relevant": bool(r.not_relevant), "w": w,
            })

    for label, (fn, map_fn) in RATED_DOMAINS.items():
        ratings, n_sup = data.clean_ratings(fn)
        suppressed[fn] = n_sup
        add_rated(ratings[ratings[SOC].isin(jobset)])
        elems = data.load(fn)[["Element ID", "Element Name"]].drop_duplicates("Element ID")
        for eid, name in elems.itertuples(index=False):
            desc = cmr.loc[eid, "Description"] if eid in cmr.index else None
            nodes.append({"node_id": eid, "label": label, "name": name, "description": desc})
            parent = eid.rsplit(".", 1)[0]
            if parent in cmr.index:
                group_ids.add(parent)
                edges.append({"src": eid, "dst": parent, "type": "IN_GROUP"})
        if map_fn:
            m = data.load(map_fn)
            id_col = [c for c in m.columns
                      if c.endswith("Element ID") and not c.startswith("Work")][0]
            pairs = m[[id_col, "Work Activities Element ID"]].drop_duplicates()
            for eid, gwa in pairs.itertuples(index=False):
                edges.append({"src": eid, "dst": gwa, "type": "ENABLES"})

    # WorkStyle nodes + ENABLES (crosswalk only; not a rated analytic domain)
    ws = data.load("work_styles_to_work_activities.csv")
    for eid, name in ws[["Work Styles Element ID", "Work Styles Element Name"]].drop_duplicates(
            "Work Styles Element ID").itertuples(index=False):
        desc = cmr.loc[eid, "Description"] if eid in cmr.index else None
        nodes.append({"node_id": eid, "label": "WorkStyle", "name": name, "description": desc})
    for eid, gwa in ws[["Work Styles Element ID", "Work Activities Element ID"]].drop_duplicates(
            ).itertuples(index=False):
        edges.append({"src": eid, "dst": gwa, "type": "ENABLES"})

    # GWA ratings: Job -[:RATED]-> GWA
    gr, n_sup = data.clean_ratings("work_activities.csv")
    suppressed["work_activities.csv"] = n_sup
    add_rated(gr[gr[SOC].isin(jobset)])

    # ---- ElementGroup nodes ----
    for gid in sorted(group_ids):
        nodes.append({"node_id": gid, "label": "ElementGroup", "name": cmr.loc[gid, "Element Name"]})

    # ---- RELATED_TO ----
    ro = data.load("related_occupations.csv")
    ro = ro[ro[SOC].isin(jobset) & ro["Related O*NET-SOC Code"].isin(jobset)]
    for r in ro[[SOC, "Related O*NET-SOC Code", "Relatedness Tier", "Index"]].itertuples(index=False):
        edges.append({"src": r[0], "dst": r[1], "type": "RELATED_TO",
                      "tier": r[2], "index": int(r[3])})

    # ---- Technology nodes + USES_TECH ----
    sw = data.load("software_skills.csv")
    sw = sw[sw[SOC].isin(jobset)]
    tech = sw.drop_duplicates("Workplace Example")[["Workplace Example", "Element Name"]]
    for example, category in tech.itertuples(index=False):
        nodes.append({"node_id": TECH_PREFIX + example, "label": "Technology",
                      "name": example, "tech_category": category})
    use = sw[[SOC, "Workplace Example", "Hot Technology", "In Demand"]].drop_duplicates(
        [SOC, "Workplace Example"])
    for soc, example, hot, dem in use.itertuples(index=False):
        edges.append({"src": soc, "dst": TECH_PREFIX + example, "type": "USES_TECH",
                      "hot_technology": hot == "Y", "in_demand": dem == "Y"})

    # ---- finalize: dedupe nodes, assign stable edge ids ----
    nodes_df = pd.DataFrame(nodes).drop_duplicates("node_id").reset_index(drop=True)
    edges_df = pd.DataFrame(edges).reset_index(drop=True)
    edges_df.insert(0, "edge_id", edges_df.index.astype("int64"))

    meta = {
        "onet_release": cfg["onet"]["release"],
        "analytic_sample": len(jobs),
        "nodes_total": int(len(nodes_df)),
        "edges_total": int(len(edges_df)),
        "nodes_by_label": nodes_df["label"].value_counts().to_dict(),
        "edges_by_type": edges_df["type"].value_counts().to_dict(),
        "suppressed_rows_dropped": suppressed,
        "rated_weight": cfg["graph"]["rated_weight"],
    }
    return nodes_df, edges_df, meta
