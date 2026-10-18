"""report: the PDF renders the model's markdown answer, not raw '##' text."""
from __future__ import annotations

import report


def test_markdown_answer_becomes_headings_bullets_and_paragraphs():
    md = ("## What Registered Nurses Do\n"
          "Nurses assess patients [N-29-1141.00]. Core tasks:\n\n"
          "- **Monitoring**: record vital signs (4.7/5) [E8386, E8388]\n"
          "- Medication \u2014 administer drugs [E8387]\n"
          "1. first\n"
          "## AI Exposure\n"
          "Not covered.")
    out = report.markdown_flowables(md)
    styles = [p.style.name for p in out]
    assert styles == ["mh", "md", "ml", "ml", "ml", "mh", "md"]
    texts = [p.text for p in out]
    assert texts[0] == "What Registered Nurses Do" and "##" not in "".join(texts)
    assert "<b>Monitoring</b>" in texts[2] and "#1D4ED8" in texts[2]   # blue citation
    assert "\u2014" in texts[3]                                       # dash kept
    assert [p.bulletText for p in out[2:5]] == ["\u2022", "\u2022", "1."]


def test_report_colors_cited_evidence_blue_not_red():
    src = open(report.__file__, encoding="utf-8").read()
    assert "b02020" not in src and "c03030" not in src
    assert report.CITED == "#1D4ED8"
