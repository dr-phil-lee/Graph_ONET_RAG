"""GraphRAG variant for chat: union of user-forced seeds and linker output,
intent-aware evidence selection, and a chat-grade grounding prompt.

Free-form chat questions often defeat the exact-match entity linker (it was
built for benchmark phrasings), so the UI/API let the caller pin occupations;
those pins come first so seed-composition heuristics see them.

The research retrieval (GraphRAG.retrieve) prioritises edges by a few
benchmark keywords ("task", "related", descriptor names). Chat questions
have broader intents — "I'm a telemarketer and want to change jobs" needs
O*NET's Related Occupations plus the profiles of those occupations and the
transferable skills that connect them. retrieve() here prepends such
intent-specific evidence so it survives the token budget.

The research SYSTEM_PROMPT (onetrag.systems.base) tells the model to "still
give your best answer" when the evidence is missing — right for a benchmark
that scores every item, wrong for a chatbot that promises no unsourced
claims. CHAT_SYSTEM_PROMPT keeps the same JSON contract but forbids outside
knowledge and asks the model to use whatever the evidence does contain.
The vendored research package is never modified: methods are overridden here.
"""
from __future__ import annotations

import re

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
from onetrag.graph import query as q
from onetrag.graph.serialize import edge_line
from onetrag.systems.graph_rag import GraphRAG

import ai_exposure

CHAT_SYSTEM_PROMPT = """You are an expert career assistant for the O*NET occupational database (release 30.3, U.S. Department of Labor). You answer ONLY from the evidence lines provided. Each line starts with its id in brackets: [E123] is one O*NET database record, [N-29-1141.00] is an occupation profile (description and Job Zone), and [X-29-1141.00] is that occupation's score in the AI Exposure Index (a third-party index from Opportunity Data, not part of O*NET).

Respond with ONLY one JSON object (no prose outside it, no markdown fences), with the keys in this order:
{"headline": <one sentence (max ~30 words) that directly answers the question>,
 "key_points": <array of up to 5 short findings, each ending with its evidence id(s), e.g. "Job Zone 2: little preparation needed [N-41-9041.00]"; [] if not useful>,
 "occupations": <array of the occupations you recommend or compare, best first, each {"soc": "<O*NET-SOC code exactly as in the evidence>", "title": "<exact O*NET title>", "fit": "<max ~25 words: why it fits or how it differs>", "citations": ["<ids>"]}; [] when the question is not about choosing or comparing occupations>,
 "answer": <string (markdown allowed), number, boolean, or array of strings: the full answer>,
 "citations": <array of every evidence id you relied on, written exactly as shown>,
 "follow_ups": <array of up to 3 short questions the user could ask next, answerable from O*NET>,
 "reasoning": <one or two sentences explaining how the cited evidence supports the answer>}

Rules:
1. Use the evidence as fully as possible. Every factual claim must be supported by a cited evidence line, and each claim should name its evidence id inline, e.g. "Customer Service Representatives [E4512]". Never use outside knowledge about O*NET, occupations, ratings, or AI exposure.
2. Copy occupation, task, skill, and activity names and all numbers (importance, level, Job Zone, relatedness rank, AI exposure score) exactly as written in the evidence. Never round, estimate, or invent values.
3. For list questions "answer" is an array of names, ordered as the evidence supports it (e.g. by importance, highest first). For numeric questions it is the number. For yes/no questions it is true or false. For open or advisory questions (career change, "what does X do", comparisons, recommendations) it is a well-organised markdown string.
4. For career-change, "similar job", "what else could I do" or recommendation questions: the "is related to the occupation" records are O*NET's official Related Occupations, and the occupation profiles and ratings describe them - that IS sufficient evidence. Recommend from those records, ordered by relatedness rank, and explain each using its profile (description, Job Zone) and the shared skills, abilities or activities in the evidence. Fill "occupations" with them.
5. AI exposure: when [X-...] lines are present, report the composite AI exposure score (a number from 0 to 1, exactly as written) of the occupations you discuss and cite those lines; for career changes, point out which related occupations have a lower score than the current one and by how much. Give scores only: never use or invent category labels for them (no "tier", "moderate", "highly exposed", "AI-protected" or similar). Do not explain how the index is computed - the app shows its methodology above your answer. If an occupation has no [X-...] line, say its AI exposure is not covered - never guess it.
6. For questions comparing two occupations, cite evidence for BOTH occupations.
7. If the evidence answers only part of the question, answer that part fully and add one sentence saying what the evidence does not cover. Only when NO evidence line is relevant to the question, start "headline" and "answer" with "Insufficient evidence:" and set "citations" to [].
8. Cite only ids that appear in the evidence. Never fabricate ids. The conversation history (if given) is NOT evidence: use it only to understand what the user refers to ("those jobs", "the second one") and never cite it.
9. Write every text field (headline, key_points, fit, answer, follow_ups, reasoning) in English only, whatever language the question is written in.
10. Be concise so the reply is never cut off: "answer" under about 300 words and do not repeat every key point in it; at most 6 occupations. The JSON must be complete and valid."""


# question-intent detectors: word-boundary regexes (so "transfer" never fires
# on "transferable skills", "path" never on "Pathologists")
_CAREER_CHANGE = re.compile(
    r"\b(chang(e|es|ed|ing)|switch(es|ed|ing)?|transition(s|ed|ing)?|move (to|into|on)|"
    r"alternative(s)?|similar (job|occupation|career|role)s?|other (job|occupation|career|role)s?|"
    r"what else|instead|recommend(s|ed|ation|ations)?|suggest(s|ed|ion|ions)?|options?|"
    r"pivot(s|ed|ing)?|(next|new|different|another) (job|career|occupation|role)|leave|quit|"
    r"related|become|could i|can i|should i|career path(s)?)\b")
_TECH = re.compile(r"\b(software|technolog(y|ies)|tools?|programs?|applications?|systems?)\b")
_OVERVIEW = re.compile(
    r"\b(what (do|does) (a|an|the)?|describe|overview|about the job|typical day|job description|"
    r"responsibilit(y|ies)|duties|do all day|tell me about)\b")
_TASKS = re.compile(r"\b(tasks?|dut(y|ies)|responsibilit(y|ies)|activit(y|ies))\b")
_COMPARE = re.compile(r"\b(compar(e|es|ed|ing|ison)|differen(t|ce|ces)|differ|vs|versus|"
                      r"between|similarit(y|ies)|better)\b")
# "ai" is matched between non-letters so "AI-driven" hits but "main" / "said"
# do not
_AI = re.compile(r"(?<![a-z])(ai|a\.i\.)(?![a-z])|"
                 r"\b(artificial intelligence|automat(e|ed|es|ion|ing)|robots?|"
                 r"chatgpt|llms?|generative|replaced|replace|future[- ]proof|exposure|exposed)\b")

# how many prior turns of conversation the generator sees, and how much of each
HISTORY_TURNS = 4
HISTORY_CHARS = 500


def _has(ql: str, rx: re.Pattern) -> bool:
    return bool(rx.search(ql))


class HybridGraphRAG(GraphRAG):
    name = "s2_graph_hybrid"
    system_prompt = CHAT_SYSTEM_PROMPT

    def seeds_for(self, item: dict) -> list[str]:
        """item["seeds"] (already resolved by ChatEngine.resolve_seeds, with
        weak alt-title links filtered out) wins outright; otherwise union the
        caller's forced seeds with the research linker's output."""
        if item.get("seeds") is not None:
            return [s for s in item["seeds"] if s in self.G][:6]
        forced = [s for s in item.get("forced_seeds") or [] if s in self.G]
        linked = super().seeds_for(item)
        return list(dict.fromkeys(forced + linked))[:6]

    # ----------------------------------------------------------- retrieval
    def retrieve(self, item: dict) -> list[dict]:
        units = super().retrieve(item)
        if not units:
            return units
        seeds = self.seeds_for(item)
        jobs = [s for s in seeds if self.G.nodes[s].get("label") == "Job"]
        if not jobs:
            return units
        question = (item.get("question_natural") or item["question_template"]).lower()
        extra = self._intent_units(question, jobs)
        if extra:
            # seed profiles first, then intent evidence, then the research ranking
            n_profiles = sum(1 for u in units if u["unit_id"].startswith("N-"))
            seen = {u["unit_id"] for u in units[:n_profiles]}
            merged = units[:n_profiles]
            for u in extra + units[n_profiles:]:
                if u["unit_id"] not in seen:
                    seen.add(u["unit_id"])
                    merged.append(u)
            units = merged
        return self._with_exposure(units)

    def _with_exposure(self, units: list[dict]) -> list[dict]:
        """Put each occupation profile's AI Exposure Index line right after
        it (X-<soc>), so the score travels with the profile through the
        token budget. Occupations the index does not cover get no line."""
        idx = ai_exposure.default_index()
        if not idx.available:
            return units
        out, seen = [], set()
        for u in units:
            out.append(u)
            uid = u["unit_id"]
            if not uid.startswith("N-"):
                continue
            soc = uid[2:]
            if soc in seen or soc not in self.G or self.G.nodes[soc].get("label") != "Job":
                continue
            x = idx.lookup(soc, self.G.nodes[soc].get("name", ""))
            if x is None:
                continue
            seen.add(soc)
            out.append({"unit_id": f"X-{soc}",
                        "text": x.evidence_text(self.G.nodes[soc].get("name", soc)),
                        "edge_ids": [], "node_ids": [soc]})
        return out

    def _node_unit(self, node: str) -> dict:
        """Research profile line, plus a note for occupations O*NET publishes
        no descriptor ratings for (graph_ext jobs), so the model says so
        instead of reading the missing ratings as "not important"."""
        unit = super()._node_unit(node)
        if (self.G.nodes[node].get("label") == "Job"
                and not q.out_edges(self.G, node, "RATED")):
            unit["text"] += (" (O*NET 30.3 publishes tasks, related occupations and "
                             "technology skills for this occupation, but no skill, "
                             "ability, knowledge or work-activity ratings.)")
        return unit

    def _comparison_units(self, jobs: list[str]) -> list[dict]:
        """Balanced evidence for comparing occupations: for each one its
        profile, top core tasks, top-rated skills / abilities / knowledge and
        hot technologies, interleaved so the token budget cuts all of them
        evenly, plus any direct Related Occupations link between them."""
        per_job: list[list[dict]] = []
        for job in jobs[:3]:
            units = [self._node_unit(job)]
            core = sorted(q.tasks_of(self.G, job, core_only=True),
                          key=lambda e: (-(e[3].get("importance") or 0), e[2]))
            if not core:
                core = sorted(q.tasks_of(self.G, job),
                              key=lambda e: (-(e[3].get("importance") or 0), e[2]))
            ranked = [self._edge_unit(e) for e in core[:6]]
            for label, k in (("TransferableSkill", 4), ("Ability", 3), ("Knowledge", 3)):
                ranked += [self._edge_unit((job, v, key, d)) for _, v, key, d in
                           self._top_rated_edges(job, label, k)]
            te = sorted(q.out_edges(self.G, job, "USES_TECH"),
                        key=lambda e: (not e[3].get("hot_technology"),
                                       not e[3].get("in_demand"), e[2]))
            ranked += [self._edge_unit(e) for e in te[:4]]
            per_job.append(units + ranked)
        out = [lst[0] for lst in per_job]                       # profiles first
        links = [e for a in jobs for e in q.out_edges(self.G, a, "RELATED_TO")
                 if e[1] in jobs]
        out += [self._edge_unit(e) for e in links]
        rest = [lst[1:] for lst in per_job]
        for i in range(max((len(r) for r in rest), default=0)):
            out += [r[i] for r in rest if i < len(r)]
        return out

    def _edge_unit(self, e: q.Edge) -> dict:
        u, v, k, d = e
        return {"unit_id": f"E{k}", "text": edge_line(self.G, e, self.anchors),
                "edge_ids": [int(k)], "node_ids": [u, v]}

    def _intent_units(self, ql: str, jobs: list[str]) -> list[dict]:
        out: list[dict] = []
        career = _has(ql, _CAREER_CHANGE)
        overview = _has(ql, _OVERVIEW)
        tech = _has(ql, _TECH)
        tasks = _has(ql, _TASKS)
        ai = _has(ql, _AI)
        single = len(jobs) == 1

        if not single and _has(ql, _COMPARE):
            out += self._comparison_units(jobs)
        if career and single:
            job = jobs[0]
            # O*NET Related Occupations (ranked), their profiles, and the
            # transferable skills / abilities that explain the transfer
            rel_edges = sorted(q.out_edges(self.G, job, "RELATED_TO"),
                               key=lambda e: (e[3].get("index") or 0, e[2]))[:10]
            out += [self._edge_unit(e) for e in rel_edges]
            # profiles (and so AI-exposure lines) of the closest related
            # jobs; an AI-motivated move gets more candidates to compare
            for _, v, _, _ in rel_edges[:8 if ai else 6]:
                if self.G.nodes[v].get("label") == "Job":
                    out.append(self._node_unit(v))
            for label, k in (("TransferableSkill", 6), ("Ability", 4), ("GWA", 0)):
                if k:
                    out += [self._edge_unit((job, v, key, d)) for _, v, key, d in
                            self._top_rated_edges(job, label, k)]
            for _, v, _, _ in rel_edges[:3]:
                out += [self._edge_unit((v, w, key, d)) for _, w, key, d in
                        self._top_rated_edges(v, "TransferableSkill", 3)]
        if not single and (career or ai):
            # several occupations in focus (e.g. a follow-up on "those jobs"):
            # their profiles carry the comparison and the exposure lines
            out += [self._node_unit(j) for j in jobs]
        if overview or (single and not career and not tasks and not tech
                        and self._domain_in_question(ql) is None):
            for job in jobs[:2]:
                core = sorted(q.tasks_of(self.G, job, core_only=True),
                              key=lambda e: -(e[3].get("importance") or 0))[:8]
                out += [self._edge_unit(e) for e in core]
                for label, k in (("TransferableSkill", 5), ("Ability", 4), ("Knowledge", 3)):
                    out += [self._edge_unit((job, v, key, d)) for _, v, key, d in
                            self._top_rated_edges(job, label, k)]
        if tasks:
            for job in jobs[:2]:
                core = sorted(q.tasks_of(self.G, job),
                              key=lambda e: -(e[3].get("importance") or 0))[:12]
                out += [self._edge_unit(e) for e in core]
        if tech:
            for job in jobs[:2]:
                te = q.out_edges(self.G, job, "USES_TECH")
                te.sort(key=lambda e: (not e[3].get("hot_technology"),
                                       not e[3].get("in_demand"), e[2]))
                out += [self._edge_unit(e) for e in te[:12]]
        return out

    def _top_rated_edges(self, job: str, label: str, k: int) -> list[q.Edge]:
        edges = [(job, v, key, d) for _, v, key, d in q.out_edges(self.G, job, "RATED")
                 if self.G.nodes[v].get("label") == label and not d.get("not_relevant")]
        edges.sort(key=lambda e: (-(e[3].get("importance") or 0), -(e[3].get("level") or 0), e[2]))
        return edges[:k]

    # ---------------------------------------------------------- generation
    @staticmethod
    def history_block(history: list[dict]) -> str:
        """Prior turns as context for references ("those jobs", "the second one").
        Each turn: {"question", "summary", "occupations": [titles]}."""
        turns = [h for h in history if h.get("question")][-HISTORY_TURNS:]
        if not turns:
            return ""
        lines = ["Conversation so far (context only - NOT evidence, never cite it):"]
        for i, h in enumerate(turns, 1):
            lines.append(f"User ({i}): {str(h['question'])[:HISTORY_CHARS]}")
            summ = str(h.get("summary") or "").strip()
            if summ:
                lines.append(f"Assistant ({i}): {summ[:HISTORY_CHARS]}")
            occ = [str(o) for o in h.get("occupations") or []][:8]
            if occ:
                lines.append(f"Occupations discussed ({i}): " + "; ".join(occ))
        return "\n".join(lines)

    def answer(self, item: dict, model: str | None = None) -> dict:
        """Same greedy budget fill as RAGSystem.answer, with the chat prompt."""
        units = self.retrieve(item)
        kept, used = [], 0
        for u in units:
            n = max(1, len(u["text"]) // 4)
            if used + n > self.budget and kept:
                break
            kept.append(u)
            used += n
        method = "estimate"
        if kept:
            try:
                used, method = self.llm.count_tokens("\n".join(u["text"] for u in kept))
            except Exception:  # noqa: BLE001 — reporting only; keep the estimate
                pass
        question = item["question_template"]
        interpreted = item.get("question_natural")
        ctx = ""
        if kept:
            ctx = "\n".join(f"[{u['unit_id']}] {u['text']}" if not u["text"].startswith("[")
                            else u["text"] for u in kept)
            user = f"Evidence:\n{ctx}\n\n"
        else:
            user = ("Evidence:\n(none — no graph entities were linked to the question)"
                    "\n\n")
        history = self.history_block(item.get("history") or [])
        if history:
            user += history + "\n\n"
        user += f"Question: {question}"
        if interpreted and interpreted.strip() != question.strip():
            user += f"\n(Read with the conversation, the question means: {interpreted})"
        resp = self.llm.complete(model=model or self.cfg["models"]["generator"],
                                 system=self.system_prompt, user=user)
        return {
            "qid": item["qid"], "system": self.name, "raw": resp["text"],
            "usage": resp["usage"], "cached": resp.get("cached", False),
            "model": resp.get("model") or getattr(self.llm, "model", None),
            "context_units": [u["unit_id"] for u in kept],
            "unit_edge_ids": {u["unit_id"]: u.get("edge_ids", []) for u in kept},
            "unit_node_ids": {u["unit_id"]: u.get("node_ids", []) for u in kept},
            "context_edge_ids": sorted({e for u in kept for e in u.get("edge_ids", [])}),
            "context_text": ctx, "context_tokens": used, "token_method": method,
        }

