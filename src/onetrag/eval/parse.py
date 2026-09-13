"""Strict-ish extraction of the JSON answer object from model output, plus
normalization of predicted strings to canonical O*NET node names/ids."""
from __future__ import annotations

import json
import re

from rapidfuzz import fuzz


def extract_json(text: str) -> dict | None:
    """First JSON object in the response; None on failure (scored as wrong)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    start = None
    return None


def norm_surface(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'().,;:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


TYPE_SUFFIX = re.compile(r"\s*\((?:task|dwa|iwa|gwa|ability|skill|knowledge(?: area)?|"
                         r"work activity|detailed[^)]*|intermediate[^)]*|general[^)]*)\)\s*$",
                         re.I)


def match_to_gold(pred: str, gold_surfaces: list[str], fuzzy_cutoff: int = 92) -> int | None:
    """Index of the gold surface this prediction names, or None."""
    p = norm_surface(TYPE_SUFFIX.sub("", pred))
    for i, g in enumerate(gold_surfaces):
        if p == norm_surface(g):
            return i
    best, best_i = 0, None
    for i, g in enumerate(gold_surfaces):
        gn = norm_surface(g)
        score = fuzz.ratio(p, gn)
        # containment (one names a prefix/suffix of the other), length-guarded
        if p and gn and min(len(p), len(gn)) / max(len(p), len(gn)) >= 0.5:
            score = max(score, fuzz.partial_ratio(p, gn))
        if score > best:
            best, best_i = score, i
    return best_i if best >= fuzzy_cutoff else None


def extract_number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return float(m.group(0))
    return None


def as_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        parts = re.split(r"\s*(?:->|→)\s*|[;\n]|,(?=\s*[A-Z])", value)
        return [p.strip() for p in parts if p.strip()]
    return [str(value)]
