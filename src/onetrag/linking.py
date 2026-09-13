"""Entity linking: question text -> graph node ids.

Tiers: exact/alias match on known surface forms -> fuzzy (rapidfuzz) over
candidate spans. An oracle mode (seed ids given) is used in ablations to
decompose linking error from reasoning error.
"""
from __future__ import annotations

import re

import networkx as nx
from rapidfuzz import fuzz, process

from .graph.store import name_index

LINKABLE_LABELS = {"Job", "Task", "EssentialSkill", "TransferableSkill", "Ability",
                   "Knowledge", "WorkStyle", "GWA", "IWA", "DWA", "Technology"}

SOC_RE = re.compile(r"\b\d{2}-\d{4}\.\d{2}\b")


class EntityLinker:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        idx = name_index(G)
        # keep only linkable labels; prefer shorter node lists (unambiguous surfaces)
        self.surface_to_ids: dict[str, list[str]] = {}
        for surface, ids in idx.items():
            keep = [i for i in ids if G.nodes[i].get("label") in LINKABLE_LABELS]
            if keep:
                self.surface_to_ids[surface] = sorted(set(keep))
        self.surfaces = list(self.surface_to_ids)

    def link(self, question: str, max_entities: int = 6,
             fuzzy_cutoff: int = 90) -> list[str]:
        ql = question.lower()
        found: dict[str, int] = {}  # node id -> position of first occurrence

        # explicit SOC codes always link
        for m in SOC_RE.finditer(question):
            if m.group(0) in self.G:
                found.setdefault(m.group(0), m.start())

        # exact surface matches, longest first; require word boundaries and
        # suppress surfaces nested inside an already-matched longer span
        # (e.g. an alt title hiding inside another occupation's full title)
        matched_spans: list[tuple[int, int]] = []
        for surface in sorted(self.surface_to_ids, key=len, reverse=True):
            if len(surface) < 4:
                continue
            pos = ql.find(surface)
            while pos >= 0:
                end = pos + len(surface)
                boundary = ((pos == 0 or not ql[pos - 1].isalnum())
                            and (end == len(ql) or not ql[end].isalnum()))
                nested = any(s <= pos and end <= e for s, e in matched_spans)
                # short single-word alt titles ("Tier", "Host") collide with
                # ordinary prose — accept only if capitalized in the question
                prose_risk = (len(surface) < 8 and " " not in surface
                              and question[pos:end] == question[pos:end].lower())
                if boundary and not nested and not prose_risk:
                    break
                pos = ql.find(surface, pos + 1)
            if pos < 0:
                continue
            matched_spans.append((pos, pos + len(surface)))
            ids = self.surface_to_ids[surface]
            if len(ids) == 1:
                found.setdefault(ids[0], pos)

        # fuzzy fallback over quoted spans (benchmark questions quote entity names)
        for span in re.findall(r"[\"']([^\"']{4,120})[\"']", question):
            hit = process.extractOne(span.lower(), self.surfaces,
                                     scorer=fuzz.ratio, score_cutoff=fuzzy_cutoff)
            if hit:
                ids = self.surface_to_ids[hit[0]]
                if len(ids) == 1:
                    found.setdefault(ids[0], ql.find(span.lower()) % 10_000)

        ordered = sorted(found.items(), key=lambda t: t[1])
        return [nid for nid, _ in ordered[:max_entities]]
