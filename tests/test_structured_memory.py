"""AI Exposure Index join, structured answers (headline / key points /
occupation cards / focus graph), conversation memory, session store, and the
API surface for all of them. Fake LLM, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ai_exposure
import answer_view
import api as api_mod
import graph_view
import memory
from tests.conftest import FakeLLM

TM = "41-9041.00"     # Telemarketers           (index: 0.6291, AI-exposed)
CSR = "43-4051.00"    # Customer Service Reps   (index: 0.6107, AI-exposed)
CRC = "41-2021.00"    # Counter and Rental Clerks (index: 0.4572, Moderate)
CEO = "11-1011.00"    # Chief Executives        (not in the index)
BIO = "15-2099.01"    # detailed O*NET code whose index row has the same title


def _with_llm(engine, llm, question, **kw):
    prev = engine.default_llm
    engine.default_llm = llm
    try:
        return engine.ask(question, **kw)
    finally:
        engine.default_llm = prev


def _structured(cards, cites=None, **extra):
    obj = {"headline": "Try Customer Service Representatives [N-43-4051.00].",
           "key_points": ["Telemarketers are AI-exposed [X-41-9041.00]"],
           "occupations": cards, "answer": "details", "citations": cites or [],
           "follow_ups": ["Which is least exposed?", "What do they do?", "x", "y"],
           "reasoning": "r"}
    obj.update(extra)
    return json.dumps(obj)


# ------------------------------------------------------------ AI exposure
def test_exposure_lookup_exact_detailed_and_missing():
    idx = ai_exposure.default_index()
    assert len(idx) == 696
    x = idx.lookup(TM, "Telemarketers")
    assert (x.composite, x.match, x.soc) == (0.6291, "exact", "41-9041")
    assert x.calculation == "(0.70 - 0.50 - 0.31 + 2) / 3 = 0.63"
    assert "tier" not in x.to_dict()
    assert x.employment == 58430
    d = idx.lookup(BIO, "Bioinformatics Technicians")
    # detailed code joined because the index row IS that occupation (same title)
    assert d is not None and d.match == "exact" and d.soc == "15-2099"
    assert idx.lookup(BIO, "Some Other Occupation") is None, "title must match"
    assert idx.lookup("15-2099.00", "Mathematical Science Occupations, All Other") is None
    assert idx.lookup(CEO, "Chief Executives") is None, "occupations missing from the index are never imputed"


def test_every_score_follows_the_published_formula():
    """composite = (digital - human contact - physical + 2) / 3 for all rows."""
    idx = ai_exposure.default_index()
    for soc, row in idx.rows.items():
        x = idx.lookup(soc + ".00", row["title"])
        assert abs((x.digital - x.human_contact - x.physical + 2) / 3 - x.composite) < 1e-4


def test_score_color_is_continuous_without_bands():
    lo, mid, hi = ai_exposure.COLOR_STOPS
    assert ai_exposure.score_color(0.0) == lo[1] and ai_exposure.score_color(1.0) == hi[1]
    assert ai_exposure.score_color(mid[0]) == mid[1]
    assert ai_exposure.score_color(0.40) not in (lo[1], mid[1], hi[1])
    assert not hasattr(ai_exposure, "tier_of") and not hasattr(ai_exposure, "TIERS")


def test_methodology_and_references():
    m = ai_exposure.methodology()
    assert m["formula"] == "AI exposure = (Digital - Human contact - Physical + 2) / 3"
    assert len(m["variables"]) == 9 and len(m["steps"]) == 4
    assert {v["dimension"] for v in m["variables"]} == {"Digital", "Human contact", "Physical"}
    urls = [r["url"] for r in m["references"]]
    assert ai_exposure.METHOD_URL in urls and ai_exposure.SOURCE_URL in urls
    assert any("onetcenter.org" in u for u in urls) and any("bls.gov" in u for u in urls)


def test_exposure_lines_follow_each_profile(engine, fake_llm):
    ans = engine.ask("I am a tele marketer and want to change to another job")
    ids = [u for u, _, _ in ans.evidence]
    assert ids[:2] == [f"N-{TM}", f"X-{TM}"]
    for i, u in enumerate(ids):
        if u.startswith("X-"):
            assert ids[i - 1] == "N-" + u[2:], "an X- line always sits right after its profile"
    text = dict((u, t) for u, t, _ in ans.evidence)[f"X-{TM}"]
    assert "0.63" in text and "(0.70 - 0.50 - 0.31 + 2) / 3" in text and "not O*NET" in text
    assert "tier" not in text and "AI-exposed" not in text


def test_no_exposure_line_for_uncovered_occupation(engine, fake_llm):
    ans = engine.ask("What does a Chief Executive do?", forced_seeds=[CEO])
    ids = [u for u, _, _ in ans.evidence]
    assert f"N-{CEO}" in ids and f"X-{CEO}" not in ids


def test_ai_question_widens_related_profiles(engine, fake_llm):
    plain = engine.ask("I am a telemarketer and want to change jobs")
    ai = engine.ask("I am a telemarketer and lost my job to AI. What else could I do?")
    n = lambda a: sum(1 for u, _, _ in a.evidence if u.startswith("N-"))  # noqa: E731
    assert n(ai) > n(plain)


def test_prompt_describes_structure_exposure_and_memory(engine, fake_llm):
    engine.ask("I am a telemarketer and want to change jobs")
    sp = fake_llm.last_system
    for key in ('"headline"', '"key_points"', '"occupations"', '"follow_ups"', "[X-"):
        assert key in sp
    assert "never guess it" in sp and "NOT evidence" in sp


# ------------------------------------------------------- structured answer
def test_cards_are_validated_and_enriched_from_the_graph(engine):
    cards = [{"soc": CSR, "title": "whatever the model wrote", "fit": "phone skills [N-43-4051.00]",
              "citations": ["N-43-4051.00"]},
             {"title": "Counter and Rental Clerks", "fit": "less exposed"},   # title only
             {"soc": CEO, "title": "Chief Executives", "fit": "not in evidence"},
             {"soc": "99-9999.99", "title": "Made Up"}]
    llm = FakeLLM(raw_override=_structured(cards, [f"N-{TM}"]))
    ans = _with_llm(engine, llm, "I am a tele marketer and want to change to another job")
    socs = [c["soc"] for c in ans.occupations]
    assert socs == [CSR, CRC], "unknown / not-in-evidence occupations are dropped"
    csr = ans.occupations[0]
    assert csr["title"] == "Customer Service Representatives"     # graph title, not model's
    assert csr["related_rank"] == 2 and csr["related_tier"] == "Primary-Short"
    assert csr["job_zone"] == 2 and csr["ai_exposure"]["composite"] == 0.6107
    assert csr["citations"] == [f"N-{CSR}"] and csr["top_skills"]
    assert ans.occupations[1]["ai_exposure"]["composite"] < 0.6107
    assert ans.exposure_method and ans.exposure_method["references"]
    d = ans.to_dict(include_evidence=False)
    keys = list(d)
    assert keys.index("ai_exposure_method") < keys.index("headline")
    assert "tier" not in json.dumps(d["ai_exposure"]) + json.dumps(d["occupations"]).replace(
        "related_tier", "")
    html = answer_view.method_html(ans)
    assert "How the AI exposure scores are calculated" in html
    assert "(0.70 - 0.50 - 0.31 + 2) / 3 = 0.63" in html and ai_exposure.METHOD_URL in html
    roles = {r["soc"]: r["role"] for r in ans.ai_exposure}
    assert roles == {TM: "current", CSR: "recommended", CRC: "recommended"}
    assert ans.headline.startswith("Try Customer Service")
    assert len(ans.follow_ups) == 3 and len(ans.key_points) == 1


def test_inline_ids_count_as_citations(engine):
    llm = FakeLLM(raw_override=_structured([], []))
    ans = _with_llm(engine, llm, "I am a tele marketer and want to change to another job")
    assert f"N-{CSR}" in ans.citations and f"X-{TM}" in ans.citations
    assert ans.status == "grounded"
    assert any(s["kind"] == "AI exposure" for s in ans.sources)


def test_truncated_reply_keeps_inline_citations(engine):
    raw = ('{"headline": "Move to Customer Service Representatives [N-43-4051.00]", '
           '"key_points": ["exposed [X-41-9041.00]"], "answer": "Details are cut he')
    ans = _with_llm(engine, FakeLLM(raw_override=raw),
                    "I am a tele marketer and want to change to another job")
    assert ans.status == "truncated"
    assert set(ans.citations) == {f"N-{CSR}", f"X-{TM}"}


def test_cards_fall_back_to_cited_related_occupations(engine):
    """No "occupations" array: cards come from the RELATED_TO records cited."""
    llm = FakeLLM(n_cite=4)        # N-TM, X-TM, then the first two RELATED_TO edges
    ans = _with_llm(engine, llm, "I am a tele marketer and want to change to another job")
    assert [c["soc"] for c in ans.occupations] == ["41-3091.00", CSR]   # O*NET ranks 1, 2
    assert all(c["source"] == "cited related occupation" for c in ans.occupations)


def test_headline_fallbacks():
    from engine import ChatEngine
    assert ChatEngine._headline({"answer": "## Title\n**Yes.** More text."}) == "Title"
    assert ChatEngine._headline({"answer": True}) == "Yes"
    assert ChatEngine._headline({"answer": 4}) == "4"
    assert ChatEngine._headline({"answer": ["a", "b"]}).startswith("2 item(s)")


def test_focus_graph_links_current_job_to_each_card(engine):
    cards = [{"soc": CSR, "fit": "x"}, {"soc": CRC, "fit": "y"}]
    llm = FakeLLM(raw_override=_structured(cards, [f"N-{TM}"]))
    ans = _with_llm(engine, llm, "I am a tele marketer and want to change to another job")
    targets = {engine.edge_index[e][1] for e in ans.focus_edge_ids}
    assert {CSR, CRC} <= targets
    assert set(ans.focus_node_ids) >= {TM, CSR, CRC}
    html = answer_view.focus_graph_html(ans, engine)
    assert "AI 0.63" in html and "ResizeObserver" in html


def test_to_dict_has_structured_fields(engine, fake_llm):
    d = engine.ask("I am a tele marketer and want to change to another job").to_dict()
    for k in ("headline", "key_points", "occupations", "ai_exposure", "follow_ups",
              "focus_edge_ids", "focus_node_ids", "standalone_question", "history_turns"):
        assert k in d
    json.dumps(d)


# ------------------------------------------------------------------ memory
def test_followup_is_rewritten_and_inherits_occupations(engine):
    first = _with_llm(engine, FakeLLM(), "I am a tele marketer and want to change jobs")
    turn = first.memory_turn()
    assert turn["seeds"][0] == TM and turn["occupations"][0] == "Telemarketers"
    llm = FakeLLM(context_reply={
        "standalone_question": "Which of Customer Service Representatives and Counter and "
                               "Rental Clerks is least exposed to AI?",
        "occupations": ["Customer Service Representatives", "Counter and Rental Clerks"]})
    ans = _with_llm(engine, llm, "Which of those is least exposed to AI?", history=[turn])
    assert llm.context_calls == 1 and llm.link_calls == 0
    assert ans.seeds == [CSR, CRC]
    assert all(e.source == "memory" for e in ans.entities)
    assert ans.standalone_question.startswith("Which of Customer Service")
    assert ans.history_turns == 1
    # generator sees the history (marked as non-evidence) and the rewrite
    assert "Conversation so far (context only - NOT evidence" in llm.last_user
    assert "the question means: Which of Customer" in llm.last_user
    assert "Telemarketers" in llm.last_context_user
    ids = [u for u, _, _ in ans.evidence]
    assert f"X-{CSR}" in ids and f"X-{CRC}" in ids


def test_followup_carries_over_when_rewrite_finds_nothing(engine):
    turn = {"question": "q", "summary": "s", "occupations": ["Telemarketers"], "seeds": [TM]}
    llm = FakeLLM(context_reply={})
    ans = _with_llm(engine, llm, "tell me more", history=[turn])
    assert ans.seeds == [TM] and ans.entities[0].source == "memory"


def test_no_history_means_no_rewrite_call(engine):
    llm = FakeLLM(link_reply=["Telemarketers"])
    _with_llm(engine, llm, "I sell things over the phone")
    assert llm.context_calls == 0 and llm.link_calls == 1


def test_new_named_occupation_skips_rewrite(engine):
    turn = {"question": "q", "summary": "s", "occupations": ["Telemarketers"], "seeds": [TM]}
    llm = FakeLLM()
    ans = _with_llm(engine, llm, "What is the Job Zone of Registered Nurses?", history=[turn])
    assert llm.context_calls == 0 and ans.seeds == ["29-1141.00"]


def test_memory_store_roundtrip_limits_and_persistence(tmp_path):
    st = memory.ConversationStore(max_sessions=2, max_turns=3, persist_dir=tmp_path)
    assert not st.valid_id("../etc") and st.valid_id(st.new_id())
    for i in range(5):
        st.append("a", {"question": f"q{i}", "seeds": [TM]})
    assert [t["question"] for t in st.get("a")] == ["q2", "q3", "q4"]
    st.append("b", {"question": "x"})
    st.append("c", {"question": "y"})
    # "a" was evicted from memory but survives on disk
    st2 = memory.ConversationStore(persist_dir=tmp_path)
    assert [t["question"] for t in st2.get("a")] == ["q2", "q3", "q4"]
    assert st2.delete("a") and st2.get("a") == [] and not st2.delete("a")
    with pytest.raises(ValueError):
        st.append("bad id!", {"question": "q"})
    assert memory.normalize_turns([{"question": ""}, "junk", {"question": "ok"}]) == [
        {"question": "ok", "summary": "", "occupations": [], "seeds": [], "cited": []}]


# ---------------------------------------------------------------- render
def test_rich_escapes_model_html_and_styles_ids():
    out = answer_view.rich("<script>x</script> **bold** [E12, X-41-9041.00]")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<b>bold</b>" in out
    assert "<span class='cite'>E12</span>" in out and "cite x'>X-41-9041.00" in out


def test_no_method_box_without_scores(engine, fake_llm):
    ans = engine.ask("What is the Job Zone of Chief Executives?")
    assert all(r["exposure"] is None for r in ans.ai_exposure)
    assert ans.exposure_method is None and answer_view.method_html(ans) == ""


def test_prompts_are_english_only_and_score_only():
    from hybrid_graph_rag import CHAT_SYSTEM_PROMPT
    assert "English only" in CHAT_SYSTEM_PROMPT
    assert "Give scores only" in CHAT_SYSTEM_PROMPT
    assert "score and tier" not in CHAT_SYSTEM_PROMPT


def test_app_sources_contain_no_hangul():
    import re
    root = Path(__file__).resolve().parents[1]
    files = [*root.glob("*.py"), *root.glob("tests/*.py"), root / "README.md"]
    for f in files:
        assert not re.search("[\uac00-\ud7a3]", f.read_text(encoding="utf-8")), f.name


def test_graph_view_colors_jobs_by_score(engine):
    edges = [engine.edge_index[e] for e in list(engine.edge_index)[:0]]
    html = graph_view.render_edges_html(
        engine.G, edges, engine.anchors, extra_nodes=[TM, CEO],
        focus_nodes={TM}, exposure={TM: engine.exposure_for(TM)},
        legend=graph_view.exposure_legend())
    assert engine.exposure_for(TM)["color"] in html and "#9CA3AF" in html
    assert "AI exposure score 0.25 or lower" in html and "Moderate" not in html


# -------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(engine):
    api_mod.app.state.engine = engine
    api_mod.app.state.sessions = memory.ConversationStore()
    with TestClient(api_mod.app) as c:
        yield c


def test_api_session_memory_flow(client, engine):
    prev = engine.default_llm
    engine.default_llm = FakeLLM(context_reply={
        "standalone_question": "Is Customer Service Representatives exposed to AI?",
        "occupations": ["Customer Service Representatives"]})
    try:
        r1 = client.post("/ask", json={"question": "I am a tele marketer and want to change "
                                                   "to another job", "session_id": "new",
                                       "include_evidence": False})
        assert r1.status_code == 200, r1.text
        b1 = r1.json()
        sid = b1["session_id"]
        assert b1["session_turns"] == 1 and b1["memory_turn"]["seeds"][0] == TM
        assert b1["headline"] and b1["ai_exposure"][0]["exposure"]["composite"] == 0.6291
        assert b1["ai_exposure_method"]["references"][0]["url"] == ai_exposure.METHOD_URL
        fg = b1["focus_graph"]
        assert any(n["id"] == TM and n["ai_exposure"]["composite"] == 0.6291 for n in fg["nodes"])
        r2 = client.post("/ask", json={"question": "is that one exposed to AI?",
                                       "session_id": sid})
        b2 = r2.json()
        assert b2["history_turns"] == 1 and b2["session_turns"] == 2
        assert b2["seeds"] == [CSR] and b2["standalone_question"].startswith("Is Customer")
        s = client.get(f"/sessions/{sid}").json()
        assert s["count"] == 2 and s["turns"][0]["seeds"][0] == TM
        assert client.delete(f"/sessions/{sid}").json()["deleted"] is True
        assert client.get(f"/sessions/{sid}").status_code == 404
        assert client.post("/ask", json={"question": "x", "session_id": "../bad"}
                           ).status_code == 422
    finally:
        engine.default_llm = prev


def test_api_stateless_history(client, engine):
    prev = engine.default_llm
    engine.default_llm = FakeLLM(context_reply={})
    try:
        hist = [{"question": "q", "summary": "s", "occupations": ["Telemarketers"],
                 "seeds": [TM]}]
        b = client.post("/ask", json={"question": "tell me more", "history": hist}).json()
        assert b["history_turns"] == 1 and b["seeds"] == [TM] and "session_id" not in b
        b = client.post("/ask", json={"question": "tell me more", "history": hist,
                                      "use_memory": False}).json()
        assert b["history_turns"] == 0
    finally:
        engine.default_llm = prev


def test_api_ai_exposure_endpoints(client):
    about = client.get("/ai-exposure").json()
    assert about["occupations"] == 696 and "tiers" not in about
    assert about["score_summary"]["max"] <= 1 and len(about["methodology"]["references"]) == 5
    one = client.get(f"/ai-exposure/{TM}").json()
    assert one["composite"] == 0.6291 and "tier" not in one
    assert client.get(f"/ai-exposure/{CEO}").status_code == 404
    assert client.get("/ai-exposure/00-0000.00").status_code == 404
    prof = client.get(f"/occupations/{TM}").json()
    assert prof["ai_exposure"]["calculation"].endswith("= 0.63")
    assert client.get(f"/occupations/{CEO}").json()["ai_exposure"] is None
