"""Offline tests of the chat engine: linking, retrieval, parsing, grounding."""
from __future__ import annotations

import json

import pytest

from engine import ChatEngine, NotConfiguredError, _MissingKeyLLM
from tests.conftest import FakeLLM

RN = "29-1141.00"      # Registered Nurses
PHARM = "29-1051.00"   # Pharmacists


# ------------------------------------------------------------ graph loaded
def test_graph_matches_meta(engine):
    meta = engine.graph_stats()
    # research graph (30,328 / 236,125 / 894) + graph_ext (776 / 4,171 / 29)
    assert engine.G.number_of_nodes() == meta["nodes_total"] == 30328 + 776
    assert engine.G.number_of_edges() == meta["edges_total"] == 236125 + 4171
    assert len(engine.occupations) == meta["analytic_sample"] == 894 + 29
    assert meta["analytic_sample_research"] == 894
    assert engine.anchors is not None, "bundled level anchors must load"


def test_src_is_frozen_copy_of_research_package():
    """The vendored onetrag package must stay byte-identical to Research/src."""
    import filecmp
    from pathlib import Path

    here = Path(__file__).resolve().parents[1]
    research = here.parent / "src" / "onetrag"
    if not research.is_dir():
        pytest.skip("Research/src not present")
    diff = []
    for p in (here / "src" / "onetrag").rglob("*.py"):
        rel = p.relative_to(here / "src" / "onetrag")
        other = research / rel
        if not other.exists() or not filecmp.cmp(p, other, shallow=False):
            diff.append(str(rel))
    assert not diff, f"vendored files differ from Research/src: {diff}"


# --------------------------------------------------------------- linking
def test_exact_linker_finds_named_occupation(engine):
    seeds = engine.preview_seeds("What is the Job Zone of Registered Nurses?")
    assert (RN, "Registered Nurses", "Job") in seeds


def test_generic_words_do_not_match_occupations(engine):
    # "zone" used to pull in Sales Managers via the alt title "Zone Manager";
    # "data" used to pull in Data Entry Keyers.
    res = engine.resolve_seeds("What is the Job Zone of Registered Nurses?")
    assert [e.node_id for e in res.entities] == [RN]
    assert res.matched == []
    assert engine.auto_occupations("Which jobs use data the most?") == []


def test_plural_free_form_words_match_head_noun(engine):
    auto = dict(engine.auto_occupations("Compare nurses and teachers"))
    assert RN in auto
    titles = list(auto.values())
    assert any(t.endswith("Teachers") or ", Postsecondary" in t for t in titles)


def test_bigram_matches_multiword_title(engine):
    sugg = engine.suggest_occupations("skills of a data analyst")
    assert "data analyst" in sugg
    # no primary title contains "data analyst": the alternate-title fallback
    # must still produce only Job nodes, deterministically ranked
    assert sugg["data analyst"]
    assert all(engine.G.nodes[n]["label"] == "Job" for n, _ in sugg["data analyst"])
    assert sugg["data analyst"] == engine.suggest_occupations("skills of a data analyst")["data analyst"]


def test_pinned_seed_is_first_and_deduplicated(engine):
    res = engine.resolve_seeds("What are the tasks of Registered Nurses?", pinned=[RN])
    assert [e.source for e in res.entities] == ["pinned"]
    assert res.seeds == [RN]


def test_search_occupations_exact_substring_fuzzy(engine):
    hits = engine.search_occupations("Registered Nurses", 5)
    assert hits[0]["soc"] == RN and hits[0]["score"] == 100
    assert any(h["soc"] == RN for h in engine.search_occupations("registered nurs", 5))
    fuzzy = engine.search_occupations("registred nurse", 5)   # typo
    assert any(h["soc"] == RN for h in fuzzy)
    assert all(h["score"] < 90 for h in fuzzy if h["matched_on"] != "title" and h["score"] < 95)
    socs = [h["soc"] for h in engine.search_occupations("nurse", 20)]
    assert len(socs) == len(set(socs)), "one row per occupation"
    assert engine.search_occupations("", 5) == []


# ------------------------------------------------------------- retrieval
def test_ask_single_job_profile_first_and_budget_respected(engine, fake_llm):
    ans = engine.ask("What is the Job Zone of Registered Nurses?", budget=1500)
    assert ans.seeds == [RN]
    assert ans.evidence[0][0] == f"N-{RN}"
    assert "Job Zone: 4" in ans.evidence[0][1]
    assert ans.context_tokens <= 1500 + 200  # greedy fill stops at unit boundary
    assert ans.status == "grounded"
    assert ans.citations == [u for u, _, c in ans.evidence if c]
    assert ans.model == "fake/model"


def test_two_job_question_uses_bridge_mode(engine, fake_llm):
    ans = engine.ask("How are Registered Nurses and Pharmacists related?")
    assert set(ans.seeds) == {RN, PHARM}
    types = {engine.edge_index[e][3]["type"] for e in ans.retrieved_edge_ids}
    assert {"HAS_TASK", "MAPS_TO", "PART_OF"} <= types


def test_domain_keyword_prioritises_rated_edges(engine, fake_llm):
    ans = engine.ask("What are the most important transferable skills for Registered Nurses?")
    # evidence order == retrieval order: after the job profile line, the
    # domain-specific RATED edges must come first
    first = [uid for uid, _, _ in ans.evidence if uid.startswith("E")][:5]
    labels = {engine.G.nodes[engine.edge_index[int(u[1:])][1]].get("label") for u in first}
    assert labels == {"TransferableSkill"}, labels


def test_no_entities_means_no_evidence(engine, fake_llm):
    ans = engine.ask("Tell me something interesting.")
    assert ans.evidence == [] and ans.seeds == []
    assert "(none" in fake_llm.last_user
    assert ans.status == "ungrounded"


def test_chat_prompt_is_used(engine, fake_llm):
    engine.ask("What is the Job Zone of Registered Nurses?")
    assert "Insufficient evidence" in fake_llm.last_system
    assert "Never use outside knowledge" in fake_llm.last_system


# ------------------------------------------------------------- grounding
def test_invalid_and_oddly_formatted_citations(engine):
    llm = FakeLLM(n_cite=0, extra_citations=["[E999999999]", "e1", "E8392, E31226", 20085])
    prev = engine.default_llm
    engine.default_llm = llm
    try:
        ans = engine.ask("What is the Job Zone of Registered Nurses?", budget=3000)
    finally:
        engine.default_llm = prev
    shown = {u for u, _, _ in ans.evidence}
    assert "E999999999" in ans.invalid_citations
    for c in ans.citations:
        assert c in shown
    assert ans.status in ("partially_grounded", "ungrounded")


def test_normalize_citations_unit():
    units = ["N-29-1141.00", "E10", "E20"]
    valid, invalid = ChatEngine._normalize_citations(
        ["[E10]", "e20", "n-29-1141.00", "E10", 99, "E10 and E20", None, True], units)
    assert valid == ["E10", "E20", "N-29-1141.00"]
    assert invalid == ["E99", "None", "True"]
    assert ChatEngine._normalize_citations(None, units) == ([], [])
    assert ChatEngine._normalize_citations("E20", units) == (["E20"], [])


def test_truncated_json_is_repaired(engine):
    raw = '{"answer": ["a", "b", "c'
    llm = FakeLLM(raw_override=raw)
    prev = engine.default_llm
    engine.default_llm = llm
    try:
        ans = engine.ask("What is the Job Zone of Registered Nurses?")
    finally:
        engine.default_llm = prev
    assert ans.parsed is not None and ans.parsed["_truncated"]
    assert ans.parsed["answer"] == ["a", "b"]
    assert ans.status == "truncated" and ans.truncated


def test_unparseable_reply(engine):
    llm = FakeLLM(raw_override="I cannot answer that.")
    prev = engine.default_llm
    engine.default_llm = llm
    try:
        ans = engine.ask("What is the Job Zone of Registered Nurses?")
    finally:
        engine.default_llm = prev
    assert ans.parsed is None and ans.status == "unparsed"
    assert ans.to_dict()["raw"] == "I cannot answer that."


def test_reply_cut_inside_answer_string_is_repaired():
    # the exact failure seen in the UI: a long markdown answer cut by the
    # output-token limit while still inside the "answer" string
    raw = ('{\n  "answer": "Based on your experience as a Telemarketer, here are '
           'several related occupations:\\n\\n1. **Customer Service Representatives** '
           '(Rank 2.0, Job Zone 2)\\n   - **Description:** Interact with customers')
    obj = ChatEngine._parse_json(raw)
    assert obj is not None and obj["_truncated"]
    assert obj["answer"].startswith("Based on your experience")
    assert obj["answer"].endswith("Interact with customers")
    # cut right after a key, inside an escape, and inside the citations list
    assert ChatEngine._parse_json('{"answer": "x", "citations": ["E1", "E2')["citations"] == ["E1"]
    assert ChatEngine._parse_json('{"answer": "a\\')["answer"] == "a"
    assert ChatEngine._parse_json('{"answer": "a", "citations":')["answer"] == "a"


def test_literal_newlines_inside_strings_are_tolerated():
    raw = '{"answer": "line one\nline two", "citations": ["E1"], "reasoning": "r"}'
    obj = ChatEngine._parse_json(raw)
    assert obj == {"answer": "line one\nline two", "citations": ["E1"], "reasoning": "r"}


def test_finish_reason_length_marks_truncated(engine):
    class Cut(FakeLLM):
        def complete(self, *, model, system, user, **kw):
            out = super().complete(model=model, system=system, user=user, **kw)
            out["usage"]["finish_reason"] = "length"
            return out

    prev = engine.default_llm
    engine.default_llm = Cut()
    try:
        ans = engine.ask("What is the Job Zone of Registered Nurses?")
    finally:
        engine.default_llm = prev
    assert ans.status == "truncated" and ans.citations


def test_fenced_json_is_parsed():
    obj = ChatEngine._parse_json('```json\n{"answer": 4, "citations": ["E1"], "reasoning": "r"}\n```')
    assert obj == {"answer": 4, "citations": ["E1"], "reasoning": "r"}


# ------------------------------------------------------------- settings
def test_missing_key_raises_not_configured(engine):
    prev = engine.default_llm
    engine.default_llm = _MissingKeyLLM()
    try:
        with pytest.raises(NotConfiguredError):
            engine.ask("What is the Job Zone of Registered Nurses?")
    finally:
        engine.default_llm = prev


def test_empty_question_rejected(engine):
    with pytest.raises(ValueError):
        engine.ask("   ")


def test_per_request_model_does_not_mutate_default(engine, fake_llm, monkeypatch):
    import engine as engine_mod

    captured = {}

    class Spy(FakeLLM):
        def __init__(self, api_key, model=None, **kw):
            super().__init__()
            captured["key"], captured["model"] = api_key, model
            self.model = model

    monkeypatch.setattr(engine_mod, "OpenRouterLLM", Spy)
    ans = engine.ask("What is the Job Zone of Registered Nurses?",
                     api_key="sk-or-test", model="x/y", budget=1000)
    assert captured == {"key": "sk-or-test", "model": "x/y"}
    assert ans.model == "x/y"
    assert engine.default_llm is fake_llm and engine.budget == 3000


def test_budget_is_clamped(engine):
    assert engine._clamp_budget(1) == 500
    assert engine._clamp_budget(10**6) == 8000
    assert engine._clamp_budget(None) == 3000


# ---------------------------------------------------------- serialisation
def test_answer_to_dict_is_json_serialisable(engine, fake_llm):
    ans = engine.ask("Which tasks of Registered Nurses involve medication?")
    d = ans.to_dict()
    json.dumps(d)
    assert d["grounded"] is True and d["evidence"][0]["id"] == f"N-{RN}"
    assert d["evidence"][0]["statement"].startswith("Occupation: Registered Nurses")
    assert d["raw"] is None
    assert d["entities"][0]["node_id"] == RN and d["entities"][0]["source"] == "linked"


def test_subgraph_and_edge_record(engine, fake_llm):
    ans = engine.ask("What is the Job Zone of Registered Nurses?")
    sg = engine.subgraph(ans.retrieved_edge_ids, set(ans.cited_edge_ids))
    assert sg["edges"] and sg["nodes"]
    assert sum(e["cited"] for e in sg["edges"]) == len(ans.cited_edge_ids)
    json.dumps(sg)
    rec = engine.edge_record(ans.retrieved_edge_ids[0])
    assert rec["statement"] and not rec["statement"].startswith("[")
    assert engine.edge_record(-1) is None


def test_node_profile_and_ego(engine):
    prof = engine.node_profile(RN)
    assert prof["label"] == "Job" and prof["attributes"]["job_zone"] == 4
    assert prof["out_edges"]["HAS_TASK"] > 10
    json.dumps(prof)
    ego = engine.ego_subgraph(RN, radius=1, edge_types={"HAS_TASK"}, max_edges=5)
    assert len(ego["edges"]) == 5 and all(e["type"] == "HAS_TASK" for e in ego["edges"])
    with pytest.raises(KeyError):
        engine.ego_subgraph("nope")
    assert engine.node_profile("nope") is None
