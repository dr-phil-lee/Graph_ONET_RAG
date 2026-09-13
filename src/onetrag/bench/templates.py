"""ONET-MHQA question templates: generator + verifier pairs.

Each template registers:
    generate(G, cfg, rng, n) -> list[item]   sampled questions with gold
    verify(G, item) -> bool                  re-derives the gold answer

Answer types: numeric (tolerance-scored), set (F1 against the full gold set),
subset (any `required` distinct gold members suffice), entity (first answer
must match), boolean.

Families:
  1-hop   T01 core tasks, T02 exact figure (contamination probe),
          T03 top-3 descriptors, T04 job zone / hot technology
  2-hop   T05 task->IWA rollup, T06 DWA-sharing neighbor, T07 skill->GWA comparison
  3-hop   T08 task->ability evidence chain, T09 dissimilar-pair IWA bridge,
          T10 transferable-skill transition overlap
  agg     T11 related-occupation superlative, T12 DWA counts
  absence T13 plausible-but-absent DWA (hallucination probe), T14 not-relevant pick
"""
from __future__ import annotations

import random
from itertools import combinations

import networkx as nx

from ..graph import query as q

DESCRIPTOR_LABELS = ["EssentialSkill", "TransferableSkill", "Ability", "Knowledge"]
LABEL_PHRASE = {"EssentialSkill": "essential skill", "TransferableSkill": "transferable skill",
                "Ability": "ability", "Knowledge": "knowledge area"}
LABEL_PLURAL = {"EssentialSkill": "essential skills", "TransferableSkill": "transferable skills",
                "Ability": "abilities", "Knowledge": "knowledge areas"}


def _jobs(G: nx.MultiDiGraph) -> list[str]:
    return sorted(n for n, d in G.nodes(data=True) if d.get("label") == "Job")


def _item(qid: str, template_id: str, hop: int, answer_type: str, question: str,
          gold: dict, evidence_edges: list[int], entities: list[str],
          strata: dict, evidence_nodes: list[str] | None = None) -> dict:
    return {"qid": qid, "template_id": template_id, "hop_count": hop,
            "answer_type": answer_type, "question_template": question,
            "question_natural": None, "gold_answer": gold,
            "gold_evidence": {"edge_ids": sorted(int(e) for e in evidence_edges),
                              "node_ids": sorted(evidence_nodes or [])},
            "entities_in_question": entities, "strata": strata}


def _strata(G, job: str) -> dict:
    return {"job_zone": G.nodes[job].get("job_zone"), "soc_major": job[:2]}


def _rated_edge(G, job: str, elem: str) -> int:
    return next(k for _, v, k, _ in q.out_edges(G, job, "RATED") if v == elem)


# ---------------------------------------------------------------------- T01

def gen_t01(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Name three core tasks of a job. Gold = all core tasks (subset scoring)."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 50:
        attempts += 1
        job = rng.choice(jobs)
        core = q.tasks_of(G, job, core_only=True)
        if len(core) < 5:
            continue
        question = (f"Name three core tasks that {q.name(G, job)} ({job}) "
                    f"perform, according to O*NET.")
        items.append(_item(
            f"T01-{len(items):05d}", "T01", 1, "subset", question,
            {"ids": [t for _, t, _, _ in core],
             "surface": [q.name(G, t) for _, t, _, _ in core],
             "value": None, "required": 3},
            [k for _, _, k, _ in core], [job], _strata(G, job)))
    return items


def verify_t01(G, item) -> bool:
    job = item["entities_in_question"][0]
    core = {t for _, t, _, _ in q.tasks_of(G, job, core_only=True)}
    return core == set(item["gold_answer"]["ids"]) and len(core) >= 5


# ---------------------------------------------------------------------- T02

def gen_t02(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Exact importance figure for a (job, element) pair — release-specific to
    two decimals, so parametric memory cannot answer."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 50:
        attempts += 1
        job = rng.choice(jobs)
        label = rng.choice(DESCRIPTOR_LABELS)
        rated = q.rated_elements(G, job, label)
        if not rated:
            continue
        elem, d = rng.choice(rated)
        if d.get("importance") is None:
            continue
        question = (f"In the O*NET 30.3 database, what is the importance rating "
                    f"(on the 1-5 scale, two decimals) of the {LABEL_PHRASE[label]} "
                    f"\"{q.name(G, elem)}\" for the occupation "
                    f"{q.name(G, job)} ({job})?")
        items.append(_item(
            f"T02-{len(items):05d}", "T02", 1, "numeric", question,
            {"value": round(float(d["importance"]), 2), "ids": [elem], "surface": []},
            [_rated_edge(G, job, elem)], [job, elem], _strata(G, job)))
    return items


def verify_t02(G, item) -> bool:
    job, elem = item["entities_in_question"]
    d = q.rating(G, job, elem)
    return d is not None and round(float(d["importance"]), 2) == item["gold_answer"]["value"]


# ---------------------------------------------------------------------- T03

def gen_t03(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Top-3 most important abilities/knowledge; rank-3 vs rank-4 gap filter
    keeps the gold unambiguous."""
    gap_min = float(cfg["benchmark"]["filters"]["topk_gap_min"])
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 80:
        attempts += 1
        job = rng.choice(jobs)
        label = rng.choice(["Ability", "Knowledge"])
        top4 = q.top_rated(G, job, label, k=4)
        if len(top4) < 4:
            continue
        if top4[2][1]["importance"] - top4[3][1]["importance"] < gap_min:
            continue
        top3 = top4[:3]
        question = (f"What are the 3 most important {LABEL_PLURAL[label]} "
                    f"(by O*NET importance rating) for {q.name(G, job)} ({job})?")
        items.append(_item(
            f"T03-{len(items):05d}", "T03", 1, "set", question,
            {"ids": [e for e, _ in top3], "surface": [q.name(G, e) for e, _ in top3],
             "value": None},
            [_rated_edge(G, job, e) for e, _ in top3], [job],
            {**_strata(G, job), "domain": label}))
    return items


def verify_t03(G, item) -> bool:
    job = item["entities_in_question"][0]
    label = item["strata"]["domain"]
    top3 = [e for e, _ in q.top_rated(G, job, label, k=3)]
    return top3 == item["gold_answer"]["ids"]


# ---------------------------------------------------------------------- T04

def gen_t04(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Alternating: (a) Job Zone (node fact), (b) hot-technology boolean."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 60:
        attempts += 1
        job = rng.choice(jobs)
        if len(items) % 2 == 0:
            jz = G.nodes[job].get("job_zone")
            if jz is None or jz != jz:
                continue
            question = (f"What Job Zone (1-5) is the occupation "
                        f"{q.name(G, job)} ({job}) assigned to in O*NET?")
            items.append(_item(
                f"T04-{len(items):05d}", "T04", 1, "numeric", question,
                {"value": float(jz), "ids": [], "surface": []},
                [], [job], {**_strata(G, job), "variant": "job_zone"},
                evidence_nodes=[job]))
        else:
            techs = q.out_edges(G, job, "USES_TECH")
            if not techs:
                continue
            _, tech, k, d = rng.choice(techs)
            gold = bool(d.get("hot_technology"))
            question = (f"Is the software \"{q.name(G, tech)}\" listed as a "
                        f"hot technology for {q.name(G, job)} ({job}) in O*NET?")
            items.append(_item(
                f"T04-{len(items):05d}", "T04", 1, "boolean", question,
                {"value": gold, "ids": [tech], "surface": []},
                [k], [job, tech], {**_strata(G, job), "variant": "hot_tech"}))
    return items


def verify_t04(G, item) -> bool:
    if item["strata"]["variant"] == "job_zone":
        job = item["entities_in_question"][0]
        return float(G.nodes[job].get("job_zone")) == item["gold_answer"]["value"]
    job, tech = item["entities_in_question"]
    for _, v, _, d in q.out_edges(G, job, "USES_TECH"):
        if v == tech:
            return bool(d.get("hot_technology")) == item["gold_answer"]["value"]
    return False


# ---------------------------------------------------------------------- T05

def gen_t05(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Task -> DWA -> IWA rollup: which IWAs does this task fall under?"""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 60:
        attempts += 1
        job = rng.choice(jobs)
        tasks = q.tasks_of(G, job)
        if not tasks:
            continue
        _, task, tk, _ = rng.choice(tasks)
        dwa_edges = q.out_edges(G, task, "MAPS_TO")
        if not dwa_edges:
            continue
        iwas, ev = set(), [tk]
        for _, dwa, dk, _ in dwa_edges:
            ev.append(dk)
            for _, iwa, pk, _ in q.out_edges(G, dwa, "PART_OF"):
                iwas.add(iwa)
                ev.append(pk)
        if not 1 <= len(iwas) <= 5:
            continue
        question = (f"In the O*NET work-activity hierarchy, the task "
                    f"\"{q.name(G, task)}\" of {q.name(G, job)} ({job}) falls under "
                    f"which intermediate work activities (IWAs)? Name all of them.")
        items.append(_item(
            f"T05-{len(items):05d}", "T05", 2, "set", question,
            {"ids": sorted(iwas), "surface": [q.name(G, i) for i in sorted(iwas)],
             "value": None},
            ev, [job, task], _strata(G, job)))
    return items


def verify_t05(G, item) -> bool:
    task = item["entities_in_question"][1]
    iwas = {i for d in q.dwas_of_task(G, task) for i in q.parents_of(G, d)}
    return sorted(iwas) == item["gold_answer"]["ids"]


# ---------------------------------------------------------------------- T06

def gen_t06(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Name another occupation whose tasks also map to a given DWA."""
    items, jobs = [], _jobs(G)
    attempts = 0
    dwas = sorted(nn for nn, dd in G.nodes(data=True) if dd.get("label") == "DWA")
    while len(items) < n and attempts < n * 80:
        attempts += 1
        job = rng.choice(jobs)
        job_dwas = sorted(q.dwas_of_job(G, job))
        if not job_dwas:
            continue
        dwa = rng.choice(job_dwas)
        others = q.jobs_with_dwa(G, dwa) - {job}
        if not 1 <= len(others) <= 15:
            continue
        ev = []
        for other in sorted(others):
            for task, _, mk, _ in q.in_edges(G, dwa, "MAPS_TO"):
                for jj, _, hk, _ in q.in_edges(G, task, "HAS_TASK"):
                    if jj == other:
                        ev += [mk, hk]
                        break
        question = (f"Besides {q.name(G, job)} ({job}), name one occupation whose "
                    f"tasks also map to the detailed work activity "
                    f"\"{q.name(G, dwa)}\" in O*NET.")
        items.append(_item(
            f"T06-{len(items):05d}", "T06", 2, "subset", question,
            {"ids": sorted(others), "surface": [q.name(G, o) for o in sorted(others)],
             "value": None, "required": 1},
            sorted(set(ev)), [job, dwa], _strata(G, job)))
    return items


def verify_t06(G, item) -> bool:
    job, dwa = item["entities_in_question"]
    others = q.jobs_with_dwa(G, dwa) - {job}
    return sorted(others) == item["gold_answer"]["ids"] and len(others) >= 1


# ---------------------------------------------------------------------- T07

def gen_t07(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Skill -> GWA -> which of two jobs rates that GWA higher (gap-filtered)."""
    gap_min = float(cfg["benchmark"]["filters"]["comparative_gap_min"])
    skills = sorted(nn for nn, dd in G.nodes(data=True)
                    if dd.get("label") in ("EssentialSkill", "TransferableSkill"))
    jobs = _jobs(G)
    items, attempts = [], 0
    while len(items) < n and attempts < n * 120:
        attempts += 1
        skill = rng.choice(skills)
        gwas = sorted(q.enabled_gwas(G, skill))
        if not gwas:
            continue
        gwa = rng.choice(gwas)
        a, b = rng.sample(jobs, 2)
        ra, rb = q.rating(G, a, gwa), q.rating(G, b, gwa)
        if not ra or not rb or ra.get("importance") is None or rb.get("importance") is None:
            continue
        if abs(ra["importance"] - rb["importance"]) < gap_min:
            continue
        winner = a if ra["importance"] > rb["importance"] else b
        enables_edge = next(k for _, v, k, _ in q.out_edges(G, skill, "ENABLES") if v == gwa)
        label = G.nodes[skill]["label"]
        question = (f"The {LABEL_PHRASE[label]} \"{q.name(G, skill)}\" enables the general "
                    f"work activity \"{q.name(G, gwa)}\" in O*NET. Which occupation rates "
                    f"that work activity as more important: {q.name(G, a)} ({a}) or "
                    f"{q.name(G, b)} ({b})?")
        items.append(_item(
            f"T07-{len(items):05d}", "T07", 2, "entity", question,
            {"ids": [winner], "surface": [q.name(G, winner)], "value": None},
            [enables_edge, _rated_edge(G, a, gwa), _rated_edge(G, b, gwa)],
            [skill, gwa, a, b], _strata(G, a)))
    return items


def verify_t07(G, item) -> bool:
    _, gwa, a, b = item["entities_in_question"]
    ra, rb = q.rating(G, a, gwa), q.rating(G, b, gwa)
    winner = a if ra["importance"] > rb["importance"] else b
    return [winner] == item["gold_answer"]["ids"]


# ---------------------------------------------------------------------- T08

def gen_t08(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Evidence chain Task -> DWA -> IWA -> GWA <- Ability. Subset scoring:
    naming the three intermediate activities of any valid path scores 1."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 80:
        attempts += 1
        job = rng.choice(jobs)
        tasks = q.tasks_of(G, job)
        if not tasks:
            continue
        _, task, tk, _ = rng.choice(tasks)
        found = None
        for elem, _ in q.top_rated(G, job, "Ability", k=10):
            paths = q.evidence_paths_task_to_element(G, task, elem)
            if paths:
                found = (elem, paths)
                break
        if not found:
            continue
        elem, paths = found
        inter = sorted({node for p in paths for node in p[1:4]})
        ev = [tk]
        for p in paths:
            _, dwa, iwa, gwa, _ = p
            ev.append(next(k for _, v, k, _ in q.out_edges(G, task, "MAPS_TO") if v == dwa))
            ev.append(next(k for _, v, k, _ in q.out_edges(G, dwa, "PART_OF") if v == iwa))
            ev.append(next(k for _, v, k, _ in q.out_edges(G, iwa, "PART_OF") if v == gwa))
            ev.append(next(k for _, v, k, _ in q.out_edges(G, elem, "ENABLES") if v == gwa))
        question = (f"In the O*NET hierarchy, the task \"{q.name(G, task)}\" of "
                    f"{q.name(G, job)} ({job}) is connected to the ability "
                    f"\"{q.name(G, elem)}\" through a chain of work activities "
                    f"(detailed -> intermediate -> general). Name the detailed, "
                    f"intermediate, and general work activity of one such chain.")
        items.append(_item(
            f"T08-{len(items):05d}", "T08", 3, "subset", question,
            {"ids": inter, "surface": [q.name(G, i) for i in inter],
             "value": None, "required": 3},
            sorted(set(ev)), [job, task, elem], _strata(G, job)))
    return items


def verify_t08(G, item) -> bool:
    _, task, elem = item["entities_in_question"]
    paths = q.evidence_paths_task_to_element(G, task, elem)
    inter = sorted({node for p in paths for node in p[1:4]})
    return bool(paths) and inter == item["gold_answer"]["ids"]


# ---------------------------------------------------------------------- T09

def _dissimilar_bridge_pairs(G, rng: random.Random, max_pairs: int):
    """Job pairs that look unrelated (different SOC major group, no O*NET
    relatedness edge, zero shared DWAs) yet share >= 1 IWA."""
    jobs = _jobs(G)
    rng.shuffle(jobs)
    related = {(u, v) for j in jobs for u, v, _, _ in q.out_edges(G, j, "RELATED_TO")}
    dwa_cache = {j: q.dwas_of_job(G, j) for j in jobs}
    iwa_cache = {j: q.iwas_of_job(G, j) for j in jobs}
    out = []
    for a, b in combinations(jobs, 2):
        if a[:2] == b[:2]:
            continue
        if (a, b) in related or (b, a) in related:
            continue
        if dwa_cache[a] & dwa_cache[b]:
            continue
        shared = iwa_cache[a] & iwa_cache[b]
        if shared and len(shared) <= 5:
            out.append((a, b, shared))
            if len(out) >= max_pairs:
                break
    return out


def _iwa_evidence(G, job: str, iwa: str) -> list[int]:
    for _, task, tk, _ in q.tasks_of(G, job):
        for _, dwa, dk, _ in q.out_edges(G, task, "MAPS_TO"):
            for _, iw, pk, _ in q.out_edges(G, dwa, "PART_OF"):
                if iw == iwa:
                    return [tk, dk, pk]
    return []


def gen_t09(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Cross-occupation bridge: shared IWAs of two superficially dissimilar jobs."""
    pairs = _dissimilar_bridge_pairs(G, rng, max_pairs=n * 3)
    rng.shuffle(pairs)
    items = []
    for a, b, shared in pairs[:n]:
        evidence = []
        for iwa in sorted(shared):
            evidence += _iwa_evidence(G, a, iwa) + _iwa_evidence(G, b, iwa)
        question = (f"The occupations {q.name(G, a)} ({a}) and {q.name(G, b)} ({b}) "
                    f"belong to different fields and are not listed as related in O*NET. "
                    f"According to the O*NET work-activity hierarchy, name every "
                    f"intermediate work activity (IWA) that BOTH occupations' tasks "
                    f"lead to (via task-to-DWA mappings and the DWA-to-IWA hierarchy).")
        items.append(_item(
            f"T09-{len(items):05d}", "T09", 3, "set", question,
            {"ids": sorted(shared), "surface": [q.name(G, i) for i in sorted(shared)],
             "value": None},
            evidence, [a, b],
            {"job_zone": G.nodes[a].get("job_zone"), "soc_major": a[:2],
             "n_shared_iwas": len(shared)}))
    return items


def verify_t09(G, item) -> bool:
    a, b = item["entities_in_question"]
    shared = q.shared_activities(G, a, b, "IWA")
    return sorted(shared) == item["gold_answer"]["ids"] and len(shared) > 0


# ---------------------------------------------------------------------- T10

def gen_t10(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Transferable skills of JobA (importance >= 4.0) that are also among the
    top-10 transferable skills of JobB."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 80:
        attempts += 1
        a, b = rng.sample(jobs, 2)
        high_a = [(e, d) for e, d in q.rated_elements(G, a, "TransferableSkill")
                  if (d.get("importance") or 0) >= 4.0]
        if not high_a:
            continue
        top_b = {e for e, _ in q.top_rated(G, b, "TransferableSkill", k=10)}
        overlap = sorted(e for e, _ in high_a if e in top_b)
        if not 1 <= len(overlap) <= 6:
            continue
        ev = [_rated_edge(G, a, e) for e in overlap] + [_rated_edge(G, b, e) for e in overlap]
        question = (f"Which transferable skills does {q.name(G, a)} ({a}) rate at "
                    f"importance 4.0 or higher that are ALSO among the top-10 "
                    f"transferable skills of {q.name(G, b)} ({b})? Name all of them.")
        items.append(_item(
            f"T10-{len(items):05d}", "T10", 3, "set", question,
            {"ids": overlap, "surface": [q.name(G, e) for e in overlap], "value": None},
            ev, [a, b], _strata(G, a)))
    return items


def verify_t10(G, item) -> bool:
    a, b = item["entities_in_question"]
    high_a = {e for e, d in q.rated_elements(G, a, "TransferableSkill")
              if (d.get("importance") or 0) >= 4.0}
    top_b = {e for e, _ in q.top_rated(G, b, "TransferableSkill", k=10)}
    return sorted(high_a & top_b) == item["gold_answer"]["ids"]


# ---------------------------------------------------------------------- T11

def gen_t11(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Among a job's Primary-Short related occupations, which has the highest
    Job Zone? Tie-filtered."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 60:
        attempts += 1
        job = rng.choice(jobs)
        rel = q.related_jobs(G, job, tiers={"Primary-Short"})
        if len(rel) < 3:
            continue
        zones = [(r, G.nodes[r].get("job_zone")) for r, _, _ in rel]
        zones = [(r, z) for r, z in zones if z is not None and z == z]
        if len(zones) < 3:
            continue
        zones.sort(key=lambda t: -t[1])
        if zones[0][1] == zones[1][1]:
            continue  # tie -> ambiguous gold
        winner = zones[0][0]
        ev = [k for _, v, k, _ in q.out_edges(G, job, "RELATED_TO")
              if v in {r for r, _ in zones}]
        question = (f"Among the occupations listed as most closely related "
                    f"(Primary-Short tier) to {q.name(G, job)} ({job}) in O*NET, "
                    f"which one has the highest Job Zone?")
        items.append(_item(
            f"T11-{len(items):05d}", "T11", 2, "entity", question,
            {"ids": [winner], "surface": [q.name(G, winner)], "value": None},
            sorted(set(ev)), [job], _strata(G, job),
            evidence_nodes=[winner]))
    return items


def verify_t11(G, item) -> bool:
    job = item["entities_in_question"][0]
    rel = q.related_jobs(G, job, tiers={"Primary-Short"})
    zones = [(r, G.nodes[r].get("job_zone")) for r, _, _ in rel]
    zones = [(r, z) for r, z in zones if z is not None and z == z]
    zones.sort(key=lambda t: -t[1])
    return [zones[0][0]] == item["gold_answer"]["ids"]


# ---------------------------------------------------------------------- T12

def gen_t12(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """How many distinct DWAs do a job's tasks map to?"""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 50:
        attempts += 1
        job = rng.choice(jobs)
        dwas = q.dwas_of_job(G, job)
        if not dwas:
            continue
        ev = [k for _, t, k, _ in q.tasks_of(G, job)]
        ev += [k for _, t, _, _ in q.tasks_of(G, job)
               for _, _, k, _ in q.out_edges(G, t, "MAPS_TO")]
        question = (f"According to O*NET, how many DISTINCT detailed work activities "
                    f"(DWAs) do the tasks of {q.name(G, job)} ({job}) map to in total?")
        items.append(_item(
            f"T12-{len(items):05d}", "T12", 2, "numeric", question,
            {"value": float(len(dwas)), "ids": [], "surface": []},
            sorted(set(ev)), [job], _strata(G, job)))
    return items


def verify_t12(G, item) -> bool:
    job = item["entities_in_question"][0]
    return float(len(q.dwas_of_job(G, job))) == item["gold_answer"]["value"]


# ---------------------------------------------------------------------- T13

def gen_t13(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Absence probe: does any task of {job} map to {DWA}? Half the items are
    plausible-but-absent (gold No: the DWA belongs to a related occupation but
    not this one), half are present (gold Yes)."""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 100:
        attempts += 1
        job = rng.choice(jobs)
        own = q.dwas_of_job(G, job)
        if not own:
            continue
        want_absent = len(items) % 2 == 0
        if want_absent:
            rel = q.related_jobs(G, job, tiers={"Primary-Short", "Primary-Long"})
            if not rel:
                continue
            other = rng.choice(rel)[0]
            candidates = sorted(q.dwas_of_job(G, other) - own)
            if not candidates:
                continue
            dwa, gold, ev_edges = rng.choice(candidates), False, []
        else:
            dwa, gold = rng.choice(sorted(own)), True
            ev_edges = []
            for _, task, tk, _ in q.tasks_of(G, job):
                for _, v, mk, _ in q.out_edges(G, task, "MAPS_TO"):
                    if v == dwa:
                        ev_edges = [tk, mk]
                        break
                if ev_edges:
                    break
        question = (f"According to O*NET, do any of the tasks of {q.name(G, job)} "
                    f"({job}) map to the detailed work activity \"{q.name(G, dwa)}\"? "
                    f"Answer yes or no.")
        items.append(_item(
            f"T13-{len(items):05d}", "T13", 2, "boolean", question,
            {"value": gold, "ids": [dwa], "surface": []},
            ev_edges, [job, dwa],
            {**_strata(G, job), "variant": "absent" if not gold else "present"},
            evidence_nodes=[] if gold else [job, dwa]))
    return items


def verify_t13(G, item) -> bool:
    job, dwa = item["entities_in_question"]
    return (dwa in q.dwas_of_job(G, job)) == item["gold_answer"]["value"]


# ---------------------------------------------------------------------- T14

def gen_t14(G, cfg, rng: random.Random, n: int) -> list[dict]:
    """Which of these four abilities is marked NOT RELEVANT for {job}?"""
    items, jobs = [], _jobs(G)
    attempts = 0
    while len(items) < n and attempts < n * 80:
        attempts += 1
        job = rng.choice(jobs)
        nr = q.not_relevant_elements(G, job, "Ability")
        if not nr:
            continue
        target = rng.choice(sorted(nr))
        relevant = [e for e, d in q.top_rated(G, job, "Ability", k=20)]
        if len(relevant) < 3:
            continue
        distractors = rng.sample(relevant, 3)
        options = distractors + [target]
        rng.shuffle(options)
        opts_text = "; ".join(q.name(G, o) for o in options)
        question = (f"For the occupation {q.name(G, job)} ({job}), which ONE of the "
                    f"following abilities is marked 'not relevant' in O*NET: {opts_text}?")
        items.append(_item(
            f"T14-{len(items):05d}", "T14", 1, "entity", question,
            {"ids": [target], "surface": [q.name(G, target)], "value": None},
            [_rated_edge(G, job, target)], [job] + options, _strata(G, job)))
    return items


def verify_t14(G, item) -> bool:
    job = item["entities_in_question"][0]
    target = item["gold_answer"]["ids"][0]
    nr = set(q.not_relevant_elements(G, job, "Ability"))
    others = set(item["entities_in_question"][1:]) - {target}
    return target in nr and not (others & nr)


# ------------------------------------------------------------------ registry

TEMPLATES: dict[str, dict] = {
    "T01": {"hop": 1, "answer_type": "subset", "generate": gen_t01, "verify": verify_t01},
    "T02": {"hop": 1, "answer_type": "numeric", "generate": gen_t02, "verify": verify_t02},
    "T03": {"hop": 1, "answer_type": "set", "generate": gen_t03, "verify": verify_t03},
    "T04": {"hop": 1, "answer_type": "mixed", "generate": gen_t04, "verify": verify_t04},
    "T05": {"hop": 2, "answer_type": "set", "generate": gen_t05, "verify": verify_t05},
    "T06": {"hop": 2, "answer_type": "subset", "generate": gen_t06, "verify": verify_t06},
    "T07": {"hop": 2, "answer_type": "entity", "generate": gen_t07, "verify": verify_t07},
    "T08": {"hop": 3, "answer_type": "subset", "generate": gen_t08, "verify": verify_t08},
    "T09": {"hop": 3, "answer_type": "set", "generate": gen_t09, "verify": verify_t09},
    "T10": {"hop": 3, "answer_type": "set", "generate": gen_t10, "verify": verify_t10},
    "T11": {"hop": 2, "answer_type": "entity", "generate": gen_t11, "verify": verify_t11},
    "T12": {"hop": 2, "answer_type": "numeric", "generate": gen_t12, "verify": verify_t12},
    "T13": {"hop": 2, "answer_type": "boolean", "generate": gen_t13, "verify": verify_t13},
    "T14": {"hop": 1, "answer_type": "entity", "generate": gen_t14, "verify": verify_t14},
}
