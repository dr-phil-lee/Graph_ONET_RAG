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
    from tests.test_structured_memory import _structured, _with_llm
    cards = [{"soc": "43-4051.00", "title": "Customer Service Representatives", "fit": "no cite"}]
    llm = FakeLLM(raw_override=_structured(cards, [f"N-{TM}"]))
    ans = _with_llm(engine, llm, "I am a tele marketer and want to change to another job")
    assert ans.unsupported_occupations == ["Customer Service Representatives"]


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
