"""User-feedback round: occupation confirmation before answering, AI exposure
only for AI / career questions (methodology last), honest grounding, bigger
evidence budget, Markdown report."""
from __future__ import annotations

import bootstrap
import hybrid_graph_rag as hg
import report
from tests.conftest import FakeLLM

TM = "41-9041.00"       # Telemarketers
RN = "29-1141.00"       # Registered Nurses


# ------------------------------------------------------------ confirmation
def test_inexact_occupation_asks_for_confirmation(engine):
    p = engine.propose_occupations("What does a professor do?")
    assert p["confirm"] is True and len(p["terms"]) == 1
    t = p["terms"][0]
    assert t["term"] == "professor" and t["via"] == "wording"
    titles = [title for _, title in t["candidates"]]
    assert titles and all(title.endswith("Postsecondary") for title in titles)
    assert len(set(soc for soc, _ in t["candidates"])) == len(t["candidates"])


def test_receptionist_offers_its_real_title_first(engine):
    p = engine.propose_occupations("I am a receptionist. What other jobs could I do?")
    assert p["confirm"] and p["terms"][0]["candidates"][0][1] == "Receptionists and Information Clerks"


def test_exact_titles_skills_and_follow_ups_need_no_confirmation(engine):
    assert not engine.propose_occupations("What does a Registered Nurse do?")["confirm"]
    assert not engine.propose_occupations("I want to be an industrial-organizational psychologist.")["confirm"]
    assert not engine.propose_occupations("What is oral expression skill?")["confirm"]
    hist = [{"question": "I am a telemarketer", "occupations": ["Telemarketers"]}]
    assert not engine.propose_occupations("Which of those is least exposed?", history=hist)["confirm"]


def test_every_name_not_written_as_its_full_title_is_confirmed(engine):
    """'zookeepers' is an alternate title (Animal Caretakers) and 'preschool
    teachers' is not the full title 'Preschool Teachers, Except Special
    Education': the card asks about both, each with its own candidates."""
    p = engine.propose_occupations(
        "Do zookeepers and preschool teachers have any work tasks in common?")
    assert p["confirm"]
    by_term = {t["term"]: t["candidates"][0] for t in p["terms"]}
    assert by_term == {"preschool teacher": ("25-2011.00", "Preschool Teachers, Except Special Education"),
                       "zookeeper": ("39-2021.00", "Animal Caretakers")}
    # full titles written out need no confirmation
    assert not engine.propose_occupations("Compare Software Developers and Data Scientists.")["confirm"]


def test_title_in_text_handles_plurals_and_qualifiers():
    from engine import title_in_text
    assert title_in_text("Registered Nurses", "What does a registered nurse do?")
    assert title_in_text("Telemarketers", "I am a telemarketer")
    assert title_in_text("zookeeper", "Do zookeepers and teachers share tasks?")
    assert not title_in_text("Preschool Teachers, Except Special Education",
                             "Do zookeepers and preschool teachers share tasks?")


def test_seeded_occupations_are_not_flagged_as_unsupported(engine, fake_llm):
    """Cards for the occupations the question was about are backed by their
    retrieved profiles even when the model left their citations empty."""
    from tests.test_structured_memory import _structured, _with_llm
    cards = [{"soc": "39-2021.00", "title": "Animal Caretakers", "fit": "x"},
             {"soc": "25-2011.00", "title": "Preschool Teachers, Except Special Education", "fit": "y"}]
    llm = FakeLLM(raw_override=_structured(cards, ["N-39-2021.00"]))
    ans = _with_llm(engine, llm, "Do zookeepers and preschool teachers share tasks?")
    assert {c["soc"] for c in ans.occupations} == {"39-2021.00", "25-2011.00"}
    assert ans.unsupported_occupations == []
    # ... and the model is told how the wording maps to O*NET titles
    assert "'zookeeper' = Animal Caretakers (39-2021.00)" in llm.last_user
    assert "'preschool teacher' = Preschool Teachers, Except Special Education" in llm.last_user


def test_unknown_occupation_is_reported_as_missing(engine):
    old = engine.default_llm
    engine.default_llm = FakeLLM(link_reply=["Glorb Whisperers"])
    try:
        p = engine.propose_occupations("What does a glorb whisperer earn?")
    finally:
        engine.default_llm = old
    assert p["confirm"] and p["terms"] == [
        {"term": "Glorb Whisperers", "candidates": [], "via": "model"}]


def test_plural_model_name_matches_the_singular_alternate_title(engine):
    hits = engine._close_occupations("Zookeepers")
    assert hits[0]["soc"] == "39-2021.00" and hits[0]["score"] >= 90


def test_model_read_occupation_is_offered_for_confirmation(engine):
    old = engine.default_llm
    engine.default_llm = FakeLLM(link_reply=["Telemarketers"])
    try:
        p = engine.propose_occupations("I sell things over the phone all day. What next?")
    finally:
        engine.default_llm = old
    assert p["confirm"] and p["terms"][0]["via"] == "model"
    assert p["terms"][0]["candidates"][0] == (TM, "Telemarketers")


def test_confirmed_occupation_replaces_the_guess(engine, fake_llm):
    guess = engine.ask("What does a professor do?")
    assert any(e.confidence == "low" for e in guess.entities)
    ans = engine.ask("What does a professor do?", forced_seeds=["25-1011.00"])   # Business Teachers
    assert ans.seeds == ["25-1011.00"]
    assert [e.node_id for e in ans.entities] == ["25-1011.00"]
    assert "📌 Business Teachers, Postsecondary" in ans.entity_caption


# -------------------------------------------------- AI exposure relevance
def test_exposure_lines_only_for_ai_or_career_questions(engine, fake_llm):
    plain = engine.ask("What does a Registered Nurse do?")
    assert not plain.exposure_in_evidence and not plain.ai_relevant
    assert plain.to_dict()["ai_exposure_method"] is None
    ai = engine.ask("How exposed are Telemarketers to AI?")   # (RN is not in the index)
    assert ai.exposure_in_evidence and ai.ai_relevant
    d = ai.to_dict(include_evidence=False)
    assert list(d)[-1] == "ai_exposure_method" and d["ai_exposure_method"]["meaning"]
    career = engine.ask("I am a telemarketer and want to change jobs")
    assert career.exposure_in_evidence          # scores help weigh the options


def test_exposure_relevance_helpers():
    assert hg.ai_question("Is my job safe from AI?") and not hg.ai_question("What do nurses do?")
    assert hg.exposure_relevant("What else could I do instead?")
    assert not hg.exposure_relevant("Which tasks involve medication?")


def test_prompt_tells_the_model_to_be_honest_about_gaps(engine, fake_llm):
    engine.ask("What does a Registered Nurse do?")
    sys_prompt = hg.CHAT_SYSTEM_PROMPT
    assert "The evidence does not cover" in sys_prompt and 'leave "occupations" empty' in sys_prompt
    assert "do not bring up AI exposure" in sys_prompt
    assert "above your answer" not in sys_prompt


# ------------------------------------------------------------- grounding
def test_unsupported_occupations_are_flagged(engine):
    """A listed occupation the question was not about, with no citation on
    its card and no cited line about it, is the model's own suggestion."""
    from tests.test_structured_memory import _structured, _with_llm
    cards = [{"soc": "43-4051.00", "title": "Customer Service Representatives", "fit": "cited in headline"},
             {"soc": "41-3011.00", "title": "Advertising Sales Agents", "fit": "no cite anywhere"}]
    llm = FakeLLM(raw_override=_structured(cards, [f"N-{TM}"]))
    ans = _with_llm(engine, llm, "I am a tele marketer and want to change to another job")
    assert ans.unsupported_occupations == ["Advertising Sales Agents"]


# ---------------------------------------------------- model-independent graph
def test_focus_graph_is_the_same_whichever_model_cites(engine):
    """The retrieved records around the occupations asked about are shown
    for every model; different citations only change the highlighting."""
    from tests.test_structured_memory import _structured, _with_llm
    q = "What does a Registered Nurse do?"
    a = _with_llm(engine, FakeLLM(raw_override=_structured([], [f"N-{RN}"])), q)
    first = a.retrieved_edge_ids[0]
    b = _with_llm(engine, FakeLLM(raw_override=_structured([], [f"E{first}"])), q)
    seed_edges = {e for e in a.retrieved_edge_ids
                  if RN in engine.edge_index[e][:2]}
    assert seed_edges and set(a.focus_edge_ids) == seed_edges    # retrieval alone
    assert a.focus_edge_ids == b.focus_edge_ids                  # same graph ...
    assert a.focus_node_ids == b.focus_node_ids == [RN]
    assert a.cited_edge_ids != b.cited_edge_ids                  # ... different highlights


def test_graph_layout_is_deterministic():
    """vis.js is given a fixed randomSeed, so identical records draw the
    same picture on every render."""
    import json
    import graph_view
    opts = json.loads(graph_view._OPTIONS)
    assert opts["layout"]["randomSeed"] == 7


def test_pivot_map_labels_do_not_overlap():
    """Occupations in the same Job Zone get stacked label positions."""
    import pandas as pd
    import answer_view as av
    df = pd.DataFrame([
        {"occupation": "Customer Service Representatives", "Job Zone": 2, "AI exposure": 0.25},
        {"occupation": "Counter and Rental Clerks", "Job Zone": 2, "AI exposure": 0.26},
        {"occupation": "Telemarketers", "Job Zone": 2, "AI exposure": 0.28},
        {"occupation": "Retail Salespersons", "Job Zone": 2, "AI exposure": 0.85},
        {"occupation": "Registered Nurses", "Job Zone": 4, "AI exposure": 0.40},
    ])
    out = av._place_labels(df)
    ys = out.set_index("occupation")["label_y"]
    close = ["Customer Service Representatives", "Counter and Rental Clerks", "Telemarketers"]
    assert len({round(ys[t], 3) for t in close}) == 3          # three distinct levels
    assert ys["Retail Salespersons"] == ys["Customer Service Representatives"]  # far right: level 0 again
    assert ys["Registered Nurses"] > 4 and ys["Telemarketers"] != ys["Registered Nurses"]
    assert all(abs(y - z) >= 0.15 for y, z in zip(ys, out.set_index("occupation")["Job Zone"]))


def test_retrieval_is_identical_whichever_model_answers(engine):
    """Paraphrases and follow-ups: two models that read the question
    differently still get the same evidence and the same graphs."""
    from tests.test_structured_memory import _with_llm
    q = "I sell things over the phone all day. What other jobs could I do?"
    a = _with_llm(engine, FakeLLM(link_reply=["Telemarketers"]), q)
    b = _with_llm(engine, FakeLLM(link_reply=["Customer Service Representatives"]), q)
    assert a.seeds == b.seeds and a.retrieved_edge_ids == b.retrieved_edge_ids
    hist = [{"question": "Compare Software Developers and Data Scientists.",
             "occupations": ["Software Developers", "Data Scientists"],
             "seeds": ["15-1252.00", "15-2051.00"], "summary": "..."}]
    q = "Which of those is more exposed to AI?"
    a = _with_llm(engine, FakeLLM(context_reply={
        "standalone_question": "Which of Software Developers and Data Scientists is more exposed to AI?",
        "occupations": ["Software Developers", "Data Scientists"]}), q, history=hist)
    b = _with_llm(engine, FakeLLM(context_reply={
        "standalone_question": "Is a Data Scientist more exposed to AI than a Software Developer?",
        "occupations": ["Data Scientists"]}), q, history=hist)
    assert a.seeds == b.seeds == ["15-1252.00", "15-2051.00"]
    assert a.retrieved_edge_ids == b.retrieved_edge_ids
    assert a.focus_edge_ids == b.focus_edge_ids and a.focus_node_ids == b.focus_node_ids
    assert a.standalone_question != b.standalone_question     # only the model's reading differs


# ---------------------------------------------------------------- budget
def test_budget_ceiling_and_reply_ceiling_raised():
    assert bootstrap.BUDGET_SLIDER_MAX == 10000
    assert bootstrap.MAX_CONTEXT_BUDGET >= 32000 and bootstrap.GEN_MAX_TOKENS >= 8000


# ------------------------------------------------------- markdown report
def test_markdown_report_has_every_section(engine, fake_llm):
    a1 = engine.ask("I am a telemarketer and lost my job to AI. What else could I do?")
    a2 = engine.ask("What does a Registered Nurse do?")
    md = report.build_markdown([a1, a2], engine, "fake", 3000)
    assert md.startswith("# GraphRAG-Based O*NET Chatbot")
    assert "## Q1. I am a telemarketer" in md and "## Q2. What does a Registered Nurse do?" in md
    assert "### Evidence provided to the model" in md and "| ✓ |" in md
    assert md.count("### About the AI exposure score") == 1          # only the AI question
    assert "`N-41-9041.00`" in md and "AI exposure = (Digital" in md


def test_pdf_report_matches_the_answer_layout(engine, fake_llm):
    """The PDF (the download in the header) carries no Key points, puts the
    AI exposure section last and only for the question about AI, and never
    prints the methodology steps."""
    a1 = engine.ask("I am a telemarketer and lost my job to AI. What else could I do?")
    a2 = engine.ask("What does a Registered Nurse do?")
    pdf = report.build_pdf([a1, a2], engine, "fake", 3000)
    assert pdf.startswith(b"%PDF")
    import pytest
    pymupdf = pytest.importorskip("pymupdf")         # text extraction only
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    assert "GraphRAG-Based O*NET Chatbot" in text and "Key points" not in text
    assert text.count("About the AI exposure score") == 1
    assert "Q2. What does a Registered Nurse do?" in text
    assert text.index("About the AI exposure score") < text.index("Q2. What does a Registered Nurse do?")
    assert text.index("Occupations") < text.index("About the AI exposure score")
