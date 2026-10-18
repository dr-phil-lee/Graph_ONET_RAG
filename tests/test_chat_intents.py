"""Chat-specific behaviour: weak-link filtering, compound/typo matching,
LLM-assisted linking, intent-aware evidence, and O*NET source records."""
from __future__ import annotations

import json

from tests.conftest import FakeLLM

TM = "41-9041.00"      # Telemarketers
MRA = "13-1161.00"     # Market Research Analysts and Marketing Specialists (alt title "Marketer")
RN = "29-1141.00"


def _types(engine, ans, n=12):
    """Evidence kinds in order; AI-exposure lines (X-) ride along with their
    occupation profile and are skipped here."""
    out = []
    for uid, _, _ in ans.evidence:
        if uid.startswith("X-"):
            continue
        out.append("N" if uid.startswith("N-") else engine.edge_index[int(uid[1:])][3]["type"])
        if len(out) >= n:
            break
    return out


def _with_llm(engine, llm, question, **kw):
    prev = engine.default_llm
    engine.default_llm = llm
    try:
        return engine.ask(question, **kw)
    finally:
        engine.default_llm = prev


# ---------------------------------------------------------------- linking
def test_lowercase_single_word_alt_title_link_is_dropped(engine):
    res = engine.resolve_seeds("I am a tele marketer and want to change to another job")
    ids = [e.node_id for e in res.entities]
    assert MRA not in ids, "'marketer' alt title must not link Market Research Analysts"
    assert ids == [TM] and res.entities[0].source == "matched"
    assert res.entities[0].term == "tele marketer"


def test_capitalised_alt_title_still_links(engine):
    res = engine.resolve_seeds("What does a Marketer do?")
    assert MRA in [e.node_id for e in res.entities]


def test_single_word_alt_title_links_when_it_is_the_whole_word(engine):
    res = engine.resolve_seeds("I'm a telemarketer, what other jobs could I do?")
    assert [e.node_id for e in res.entities] == [TM]


def test_typo_is_fuzzy_matched(engine):
    res = engine.resolve_seeds("skills of registred nurses")
    assert RN in [e.node_id for e in res.entities]


def test_paraphrase_without_job_word_matches_nothing(engine):
    res = engine.resolve_seeds("I sell things over the phone, what should I do next?")
    assert [e for e in res.entities if e.label == "Job"] == []


def test_low_confidence_alt_title_match_is_replaced_by_llm_link(engine):
    # "doctor" only exists as alternate titles -> low confidence guess;
    # the model's reading ("Physicians"-like title) must replace it
    res = engine.resolve_seeds("I work as a doctor and want a change")
    assert res.entities and res.entities[0].confidence == "low"
    llm = FakeLLM(link_reply=["Family Medicine Physicians"])
    ans = _with_llm(engine, llm, "I work as a doctor and want a change")
    assert llm.link_calls == 1
    assert [e.source for e in ans.entities] == ["inferred"]
    assert ans.entities[0].name == "Family Medicine Physicians"


def test_llm_link_runs_only_when_needed(engine):
    llm = FakeLLM(link_reply=["Telemarketers"])
    ans = _with_llm(engine, llm, "I sell things over the phone, what should I do next?")
    assert llm.link_calls == 1
    assert [(e.node_id, e.source, e.term) for e in ans.entities] == [(TM, "inferred", "Telemarketers")]
    assert ans.seeds == [TM]

    llm = FakeLLM(link_reply=["Telemarketers"])
    _with_llm(engine, llm, "What is the Job Zone of Registered Nurses?")
    assert llm.link_calls == 0, "exact link found: no extra LLM call"


def test_paraphrased_job_uses_llm_link(engine):
    llm = FakeLLM(link_reply=["Xqzv Wplk", "Telemarketer"])
    ans = _with_llm(engine, llm, "I cold-call strangers all day and want to change jobs")
    assert llm.link_calls == 1
    assert [e.node_id for e in ans.entities] == [TM]     # garbage name skipped, title matched
    assert ans.status == "grounded"


def test_llm_link_ignores_low_confidence_and_garbage(engine):
    llm = FakeLLM(link_reply=["Chief Wizard of Nothing", 42, ""])
    ans = _with_llm(engine, llm, "I do magic for a living, what next?")
    assert ans.entities == [] and ans.evidence == []


def test_llm_link_failure_is_tolerated(engine):
    class Broken(FakeLLM):
        def complete(self, *, model, system, user, **kw):
            if system.startswith("Extract"):
                raise RuntimeError("boom")
            return super().complete(model=model, system=system, user=user, **kw)

    ans = _with_llm(engine, Broken(), "I sell things over the phone")
    assert ans.entities == [] and ans.status in ("ungrounded", "grounded")


def test_auto_link_false_skips_llm_link(engine):
    llm = FakeLLM(link_reply=["Telemarketers"])
    ans = _with_llm(engine, llm, "I sell things over the phone", auto_link=False)
    assert llm.link_calls == 0 and ans.entities == []


# --------------------------------------------------------- intent evidence
def test_career_change_question_gets_related_occupations_first(engine, fake_llm):
    ans = engine.ask("I am a tele marketer and want to change to another job")
    assert ans.seeds == [TM], "only Telemarketers may seed (no Market Research Analysts)"
    types = _types(engine, ans, 20)
    assert types[0] == "N" and types[1:11] == ["RELATED_TO"] * 10
    # profiles of the related occupations follow, then transferable skills
    assert types[11:17] == ["N"] * 6
    assert "RATED" in types[17:]
    no_x = [u for u, _, _ in ans.evidence if not u.startswith("X-")]
    rel = [engine.edge_index[int(u[1:])] for u in no_x[1:11]]
    ranks = [d.get("index") for _, _, _, d in rel]
    assert ranks == sorted(ranks)
    # O*NET rank 1 (a graph_ext occupation), then Customer Service Representatives
    assert rel[0][1] == "41-3091.00" and ranks[0] == 1
    assert engine.G.nodes[rel[1][1]]["name"] == "Customer Service Representatives"


def test_overview_question_gets_core_tasks_and_top_ratings(engine, fake_llm):
    ans = engine.ask("What does a Registered Nurse do?")
    types = _types(engine, ans, 25)
    assert types[0] == "N"
    assert types[1:9] == ["HAS_TASK"] * 8
    assert "RATED" in types[9:]


def test_technology_question_gets_uses_tech(engine, fake_llm):
    ans = engine.ask("What software do Registered Nurses use?")
    types = _types(engine, ans, 14)
    assert "USES_TECH" in types[1:13]


def test_two_job_question_keeps_bridge_mode(engine, fake_llm):
    ans = engine.ask("Could a Registered Nurse become a Pharmacist?")
    assert set(ans.seeds) == {RN, "29-1051.00"}
    types = set(_types(engine, ans, 40))
    assert {"HAS_TASK", "MAPS_TO", "PART_OF"} <= types


def test_prompt_allows_partial_answers_and_related_occupations(engine, fake_llm):
    engine.ask("I am a telemarketer and want to change jobs")
    sp = fake_llm.last_system
    assert "Related Occupations" in sp and "sufficient evidence" in sp
    assert "answer that part fully" in sp
    assert "inline" in sp


# ----------------------------------------------------------------- sources
def test_sources_map_to_onet_tables(engine):
    llm = FakeLLM(n_cite=14)
    ans = _with_llm(engine, llm, "I'm a telemarketer, what other jobs could I do?")
    assert len(ans.sources) == len(ans.citations) == 14
    by_kind = {s["kind"]: s for s in ans.sources}
    assert by_kind["occupation profile"]["table"].startswith("Occupation Data")
    assert "related_occupations.csv" in by_kind["RELATED_TO"]["table"]
    assert all(s["dataset"] == "O*NET 30.3" for s in ans.sources
               if not s["id"].startswith("X-"))
    x = [s for s in ans.sources if s["id"].startswith("X-")]
    assert x and x[0]["kind"] == "AI exposure" and "Opportunity Data" in x[0]["dataset"]
    assert all(s["statement"] and not s["statement"].startswith("[") for s in ans.sources)
    json.dumps(ans.to_dict())
    assert ans.to_dict()["sources"][0]["id"] == f"N-{TM}"


def test_source_record_for_rated_edge(engine):
    from onetrag.graph import query as q

    v, d = q.top_rated(engine.G, RN, "TransferableSkill", 1)[0]
    eid = next(k for _, w, k, dd in engine.G.out_edges(RN, keys=True, data=True)
               if w == v and dd.get("type") == "RATED")
    rec = engine.source_record(f"E{eid}")
    assert rec["kind"] == "RATED" and "transferable_skills.csv" in rec["table"]
    assert rec["nodes"][1]["name"] == engine.G.nodes[v]["name"]
    assert rec["statement"].startswith("Registered Nurses rates")
    assert engine.source_record("E-1")["kind"] == "unknown"


def _jobs(engine, question):
    return [e.name for e in engine.resolve_seeds(question).entities]


def test_three_word_title_is_matched_as_one_phrase(engine):
    """'elementary school teachers' must not also pull in Special Education
    Teachers, Elementary School (from the pair 'elementary school')."""
    assert _jobs(engine, "Compare zookeepers and elementary school teachers.") == [
        "Elementary School Teachers, Except Special Education", "Animal Caretakers"]
    assert _jobs(engine, "I'm an elementary school teacher") == [
        "Elementary School Teachers, Except Special Education"]


def test_common_first_word_of_a_three_word_alt_title(engine):
    """'high' is a common word, but 'high school teacher' is an O*NET
    alternate title of Secondary School Teachers."""
    assert _jobs(engine, "I'm a high school teacher") == [
        "Secondary School Teachers, Except Special and Career/Technical Education"]


def test_three_word_match_consumes_positions_not_words(engine):
    assert _jobs(engine, "heavy truck driver vs bus driver") == [
        "Heavy and Tractor-Trailer Truck Drivers", "Bus Drivers, School"]
    assert _jobs(engine, "I'm a computer systems analyst") == ["Computer Systems Analysts"]
