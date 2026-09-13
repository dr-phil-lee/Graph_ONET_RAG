"""Hallucination / faithfulness measurement.

Two tiers, per the study design:
1. An LLM judge (a different model from the generator) decomposes the answer
   into atomic claims and, where a claim is a checkable O*NET assertion,
   emits it as a structured triple {occupation, element, relation, value}.
2. Every structured triple is then verified DETERMINISTICALLY against the
   knowledge graph — the headline KB-false rate never trusts the judge's own
   truth opinion, only its claim extraction.

The judge also labels whether each claim is supported by the provided context
(secondary, judge-opinion metric).
"""
from __future__ import annotations

import json

import networkx as nx
from rapidfuzz import fuzz, process

from ..graph import query as q
from ..llm.client import LLM
from .parse import extract_json, norm_surface

JUDGE_SYSTEM = """You are auditing an answer about the O*NET occupational database.
Decompose the ANSWER into atomic factual claims. Respond with ONLY JSON:
{"claims": [{"text": <claim as one sentence>,
             "occupation": <occupation name if the claim is about one, else null>,
             "element": <skill/ability/knowledge/activity/task name if any, else null>,
             "relation": <"rating"|"has_task"|"maps_to"|"related"|"other">,
             "value": <numeric rating if the claim states one, else null>,
             "supported_by_context": <true|false — is it stated in the CONTEXT?>}]}
Only include claims the answer actually asserts. Max 10 claims."""


class KBChecker:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self.job_names = {norm_surface(d["name"]): n for n, d in G.nodes(data=True)
                          if d.get("label") == "Job"}
        self.elem_names = {norm_surface(d["name"]): n for n, d in G.nodes(data=True)
                           if d.get("label") in ("EssentialSkill", "TransferableSkill",
                                                 "Ability", "Knowledge", "WorkStyle",
                                                 "GWA", "IWA", "DWA")}

    def _resolve(self, name: str | None, table: dict) -> str | None:
        if not name:
            return None
        key = norm_surface(name)
        if key in table:
            return table[key]
        hit = process.extractOne(key, list(table), scorer=fuzz.ratio, score_cutoff=92)
        return table[hit[0]] if hit else None

    def check(self, claim: dict) -> str:
        """'true' / 'false' / 'unverifiable' against the graph."""
        job = self._resolve(claim.get("occupation"), self.job_names)
        elem = self._resolve(claim.get("element"), self.elem_names)
        rel = claim.get("relation")
        if rel == "rating" and job and elem:
            d = q.rating(self.G, job, elem)
            if d is None or d.get("importance") is None:
                return "false"
            v = claim.get("value")
            if v is None:
                return "true"  # asserted the rating exists — it does
            try:
                return "true" if abs(float(v) - float(d["importance"])) <= 0.05 else "false"
            except (TypeError, ValueError):
                return "unverifiable"
        if rel == "maps_to" and job and elem:
            label = self.G.nodes[elem].get("label")
            reach = {"DWA": q.dwas_of_job, "IWA": q.iwas_of_job,
                     "GWA": q.gwas_of_job_via_tasks}.get(label)
            if reach is None:
                return "unverifiable"
            return "true" if elem in reach(self.G, job) else "false"
        if rel == "related" and job:
            other = self._resolve(claim.get("element"), self.job_names)
            if other:
                rel_jobs = {r for r, _, _ in q.related_jobs(self.G, job)}
                return "true" if other in rel_jobs else "false"
        return "unverifiable"


def judge_response(item: dict, response: dict, llm: LLM, judge_model: str,
                   checker: KBChecker) -> dict:
    obj = extract_json(response["raw"]) or {}
    answer_text = json.dumps({"answer": obj.get("answer"),
                              "reasoning": obj.get("reasoning")}, ensure_ascii=False)
    context = response.get("context_text") or ""
    user = f"CONTEXT:\n{context or '(no context given)'}\n\nANSWER:\n{answer_text}"
    resp = llm.complete(model=judge_model, system=JUDGE_SYSTEM, user=user, max_tokens=1500)
    parsed = extract_json(resp["text"]) or {"claims": []}
    claims = parsed.get("claims") or []

    verdicts = [checker.check(c) for c in claims]
    n_checkable = sum(v != "unverifiable" for v in verdicts)
    n_false = sum(v == "false" for v in verdicts)
    return {
        "qid": item["qid"], "system": response["system"],
        "n_claims": len(claims), "n_checkable": n_checkable, "n_kb_false": n_false,
        "kb_false_any": n_false > 0,
        "judge_unsupported": sum(1 for c in claims if c.get("supported_by_context") is False),
        "claims": claims, "verdicts": verdicts,
    }
