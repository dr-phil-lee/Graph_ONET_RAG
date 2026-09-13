"""Deterministic scoring of system responses against programmatic gold.

score_item() returns per-question records with the primary score in [0, 1]
plus citation precision/recall against the gold evidence edge set.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .parse import (TYPE_SUFFIX, as_string_list, extract_json, extract_number,
                    match_to_gold, norm_surface)

NUMERIC_STRICT_TOL = 0.05
NUMERIC_RELAXED_PCT = 0.10


def score_item(item: dict, response: dict) -> dict:
    obj = extract_json(response["raw"])
    rec = {"qid": item["qid"], "system": response["system"],
           "template_id": item["template_id"], "hop_count": item["hop_count"],
           "answer_type": item["answer_type"], "parse_ok": obj is not None,
           "score": 0.0, "strict": 0.0, "relaxed": 0.0,
           "citation_precision": None, "citation_recall": None,
           "context_gold_recall": _context_gold_recall(item, response)}
    if obj is None:
        return rec

    answer = obj.get("answer")
    at = item["answer_type"]
    gold = item["gold_answer"]

    if at == "numeric":
        pred = extract_number(answer)
        if pred is not None:
            diff = abs(pred - float(gold["value"]))
            rec["strict"] = float(diff <= NUMERIC_STRICT_TOL)
            rel_tol = max(NUMERIC_RELAXED_PCT * abs(float(gold["value"])), NUMERIC_STRICT_TOL)
            rec["relaxed"] = float(diff <= rel_tol)
        rec["score"] = rec["strict"]
    elif at == "set":
        preds = as_string_list(answer)
        matched = set()
        for p in preds:
            hit = match_to_gold(p, gold["surface"])
            if hit is not None:
                matched.add(hit)
        precision = len(matched) / len(preds) if preds else 0.0
        recall = len(matched) / len(gold["surface"]) if gold["surface"] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rec.update(score=f1, strict=float(recall == 1.0 and precision == 1.0), relaxed=f1)
    elif at == "subset":
        # any `required` distinct gold members suffice; wrong members cost
        # precision — except restatements of entities the question itself names
        preds = as_string_list(answer)
        required = int(gold.get("required", 1))
        question = norm_surface(item.get("question_natural") or item["question_template"])
        matched, counted = set(), 0
        for p in preds:
            hit = match_to_gold(p, gold["surface"])
            if hit is not None:
                matched.add(hit)
                counted += 1
                continue
            pn = norm_surface(TYPE_SUFFIX.sub("", p))
            if pn and fuzz.partial_ratio(pn, question) >= 90:
                continue  # echoes the question's own entity — not an error
            counted += 1
        precision = len(matched) / counted if counted else 0.0
        quota = min(len(matched), required) / required
        rec["score"] = rec["relaxed"] = precision * quota
        rec["strict"] = float(len(matched) >= required and len(matched) == counted)
    elif at == "boolean":
        pred = answer if isinstance(answer, bool) else str(answer).strip().lower() in (
            "yes", "true", "y")
        rec["score"] = rec["strict"] = rec["relaxed"] = float(pred == gold["value"])
    elif at == "entity":
        # whole string first — occupation titles contain commas that the
        # list-splitting heuristic would sever
        hit = (isinstance(answer, str)
               and match_to_gold(answer, gold["surface"]) is not None)
        if not hit:
            preds = as_string_list(answer)
            hit = bool(preds) and match_to_gold(preds[0], gold["surface"]) is not None
        rec["score"] = rec["strict"] = rec["relaxed"] = float(hit)

    rec.update(_citation_scores(item, response, obj))
    return rec


def _citation_scores(item: dict, response: dict, obj: dict) -> dict:
    """Unit-level citation precision + edge-level recall of gold evidence.

    A citation is correct if it resolves to gold evidence: a bare edge id in
    the gold edge set, or a context unit whose provenance (edge_ids/node_ids)
    intersects the gold evidence. Node evidence covers facts that live on
    nodes rather than edges (e.g. Job Zone)."""
    cited = obj.get("citations") or []
    if not isinstance(cited, list):
        cited = [str(cited)]
    gold_edges = set(item["gold_evidence"].get("edge_ids", []))
    gold_nodes = set(item["gold_evidence"].get("node_ids", []))
    has_gold = bool(gold_edges or gold_nodes)
    if not cited:
        return {"citation_precision": None,
                "citation_recall": 0.0 if gold_edges else None}

    unit_edges = {u: set(int(e) for e in es)
                  for u, es in (response.get("unit_edge_ids") or {}).items()}
    unit_nodes = {u: set(ns) for u, ns in (response.get("unit_node_ids") or {}).items()}

    correct, covered_edges = 0, set()
    for c in cited:
        c = str(c).strip().strip("[]")
        m = re.fullmatch(r"E(\d+)", c)
        edges = {int(m.group(1))} if m else unit_edges.get(c, set())
        nodes = unit_nodes.get(c, set()) if not m else set()
        if (edges & gold_edges) or (nodes & gold_nodes):
            correct += 1
            covered_edges |= edges & gold_edges
    precision = correct / len(cited)
    recall = (len(covered_edges) / len(gold_edges)) if gold_edges else None
    return {"citation_precision": precision if has_gold else None,
            "citation_recall": recall}


def _context_gold_recall(item: dict, response: dict) -> float | None:
    """Was the gold evidence even retrieved? Separates retrieval from reasoning."""
    gold = set(item["gold_evidence"]["edge_ids"])
    if not gold:
        return None
    ctx = set(response.get("context_edge_ids") or [])
    return len(gold & ctx) / len(gold)
