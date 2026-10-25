"""PDF report of chat results, with a clean title page.

build_pdf(answers, ...) -> bytes for st.download_button. Evidence subgraphs
are rendered as static matplotlib images (pyvis is JS-only, so the PDF gets
its own drawing of the same edges).
"""
from __future__ import annotations

import datetime
import re
from io import BytesIO

import bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, PolyLine
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import ai_exposure
from graph_view import LABEL_COLORS

TITLE = "GraphRAG-Based O*NET Chatbot"
SUBTITLE = "Results Report"
NAVY = colors.HexColor("#10304F")
ACCENT = colors.HexColor("#1D4ED8")
CITED = "#1D4ED8"            # cited evidence (blue, as in the app)
CITED_BG = colors.HexColor("#EAF1FF")

_SS = getSampleStyleSheet()
_STYLES = {
    "title": ParagraphStyle("t", parent=_SS["Title"], fontSize=26, leading=32,
                            textColor=NAVY, alignment=1),
    "subtitle": ParagraphStyle("st", parent=_SS["Normal"], fontSize=15,
                               leading=20, alignment=1, textColor=ACCENT,
                               fontName="Helvetica-Bold"),
    "tagline": ParagraphStyle("tg", parent=_SS["Normal"], fontSize=10.5,
                              leading=16, alignment=1,
                              textColor=colors.HexColor("#5B7290")),
    "meta": ParagraphStyle("m", parent=_SS["Normal"], fontSize=9.5,
                           leading=13, alignment=1, textColor=colors.grey),
    "q": ParagraphStyle("q", parent=_SS["Heading2"], fontSize=13, leading=17,
                        spaceBefore=4, textColor=NAVY),
    "h": ParagraphStyle("h", parent=_SS["Heading4"], fontSize=10.5,
                        spaceBefore=8, spaceAfter=2, textColor=ACCENT),
    "body": ParagraphStyle("b", parent=_SS["Normal"], fontSize=10, leading=14),
    "answer": ParagraphStyle("a", parent=_SS["Normal"], fontSize=11.5,
                             leading=16, fontName="Helvetica-Bold"),
    "ev": ParagraphStyle("e", parent=_SS["Normal"], fontSize=8, leading=10.5),
    "evc": ParagraphStyle("ec", parent=_SS["Normal"], fontSize=8, leading=10.5,
                          fontName="Helvetica-Bold",
                          textColor=colors.HexColor(CITED)),
    "mdh": ParagraphStyle("mh", parent=_SS["Normal"], fontSize=11.5, leading=15,
                          fontName="Helvetica-Bold", textColor=NAVY,
                          spaceBefore=7, spaceAfter=3),
    "md": ParagraphStyle("md", parent=_SS["Normal"], fontSize=10, leading=14.5,
                         spaceAfter=4),
    "mdli": ParagraphStyle("ml", parent=_SS["Normal"], fontSize=10, leading=14.5,
                           leftIndent=16, bulletIndent=5, spaceAfter=2.5),
    "note": ParagraphStyle("n", parent=_SS["Normal"], fontSize=8.5,
                           leading=11, textColor=colors.grey),
}


def _esc(s) -> str:
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # the built-in PDF fonts use the Windows-1252 set (dashes, quotes, bullets)
    return s.encode("cp1252", "replace").decode("cp1252")


_CIT = re.compile(r"\[\s*((?:E\d+|[NXD]-[0-9A-Za-z.\-]+)"
                  r"(?:\s*,\s*(?:E\d+|[NXD]-[0-9A-Za-z.\-]+))*)\s*\]")


def _check_mark() -> Drawing:
    """A blue check mark drawn as vector lines (no font glyph needed)."""
    d = Drawing(10, 9)
    d.add(PolyLine([1, 4.5, 4, 1.5, 9, 8], strokeColor=colors.HexColor(CITED),
                   strokeWidth=1.8, strokeLineCap=1, strokeLineJoin=1))
    return d


def _inline(text: str) -> str:
    """Escaped text with **bold**, *italic* and blue citation ids."""
    s = _esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return _CIT.sub(lambda m: "<font name='Courier' size='8' color='%s'>[%s]</font>"
                    % (CITED, m.group(1)), s)


def markdown_flowables(text: str) -> list:
    """The model's markdown answer as PDF paragraphs: headings, bullet and
    numbered lists and plain paragraphs (blank lines separate them)."""
    S = _STYLES
    out, para = [], []

    def flush():
        if para:
            out.append(Paragraph(_inline(" ".join(para)), S["md"]))
            para.clear()

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        h = re.match(r"#{1,6}\s+(.*)", line)
        b = re.match(r"[-*+•]\s+(.*)", line)
        n = re.match(r"(\d+)[.)]\s+(.*)", line)
        if h:
            flush()
            out.append(Paragraph(_inline(h.group(1).strip("*# ")), S["mdh"]))
        elif b:
            flush()
            out.append(Paragraph(_inline(b.group(1)), S["mdli"], bulletText="•"))
        elif n:
            flush()
            out.append(Paragraph(_inline(n.group(2)), S["mdli"],
                                 bulletText=f"{n.group(1)}."))
        else:
            para.append(line)
    flush()
    return out


def subgraph_png(G, edges, cited_edge_ids: set[int], max_edges: int = 60) -> bytes:
    """Static drawing of the evidence subgraph; cited edges blue and thick."""
    edges = (sorted(edges, key=lambda e: e[2] not in cited_edge_ids)[:max_edges]
             if len(edges) > max_edges else edges)
    H = nx.MultiDiGraph()
    for u, v, k, d in edges:
        H.add_edge(u, v, key=k)
    pos = nx.spring_layout(H, k=0.9 / max(1, len(H) ** 0.5), seed=7, iterations=60)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    labels_present = []
    for n in H.nodes:
        a = G.nodes[n] if n in G else {}
        lbl = a.get("label", "?")
        if lbl not in labels_present:
            labels_present.append(lbl)
        big = lbl == "Job"
        ax.scatter(*pos[n], s=260 if big else 60,
                   color=LABEL_COLORS.get(lbl, "#999999"), zorder=3,
                   edgecolors="white", linewidths=0.8)
        if big:
            name = a.get("name", n)
            ax.annotate(name if len(name) <= 34 else name[:33] + "…", pos[n],
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=7.5, fontweight="bold", zorder=4)
    for u, v, k, d in edges:
        cited = k in cited_edge_ids
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=CITED if cited else "#c9c9c9",
                lw=1.8 if cited else 0.7, alpha=0.95 if cited else 0.65,
                zorder=2 if cited else 1)
    handles = [plt.Line2D([], [], marker="o", ls="", color=LABEL_COLORS.get(l, "#999"),
                          label=l, markersize=6) for l in labels_present]
    handles.append(plt.Line2D([], [], color=CITED, lw=2, label="cited edge"))
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=False,
              ncol=min(4, len(handles)))
    ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _answer_flowables(i: int, ans, engine) -> list:
    S = _STYLES
    out = [Paragraph(f"Q{i}. {_esc(ans.question)}", S["q"])]

    if ans.parsed is None:
        out += [Paragraph("The model reply was not valid JSON:", S["body"]),
                Paragraph(_esc(ans.raw[:1500]), S["ev"])]
        return out

    if getattr(ans, "headline", ""):
        out += [Paragraph("Answer", S["h"]), Paragraph(_inline(ans.headline), S["answer"])]
    if ans.status == "ungrounded" or not ans.citations:
        out.append(Paragraph(
            "<b>Not based on O*NET evidence.</b> The model cited no retrieved record, so "
            "this answer (including any occupations it names) comes from the model itself "
            "and may not be relevant.", S["note"]))
    elif getattr(ans, "unsupported_occupations", None):
        out.append(Paragraph("<b>Listed without any cited evidence:</b> " +
                             _esc(", ".join(ans.unsupported_occupations)) + ".", S["note"]))
    rows_x = _occupation_rows(ans)
    if rows_x:
        out.append(Paragraph("Occupations", S["h"]))
        tbl = Table(rows_x, colWidths=[2.15 * inch, 0.85 * inch, 0.45 * inch, 0.6 * inch,
                                       0.6 * inch, 1.85 * inch], repeatRows=1)
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        out.append(tbl)
        out.append(Paragraph(
            _esc(f"AI exposure: {ai_exposure.SOURCE_CITATION}. {ai_exposure.SOURCE_NOTE} "
                 "Score 0 = fully protected, 1 = fully exposed."), S["note"]))

    a = ans.parsed.get("answer")
    out.append(Paragraph("Full answer", S["h"]))
    if isinstance(a, list):
        for x in a:
            out.append(Paragraph(_inline(x), S["mdli"], bulletText="•"))
    elif isinstance(a, bool):
        out.append(Paragraph("Yes" if a else "No", S["answer"]))
    elif a is not None:
        out += markdown_flowables(a)
    if ans.parsed.get("reasoning"):
        out += [Paragraph("Reasoning", S["h"]),
                Paragraph(_esc(ans.parsed["reasoning"]), S["body"])]
    if ans.seeds:
        names = ", ".join(f"{engine.node_info(s)[0]} ({s})" for s in ans.seeds)
        out += [Paragraph("Graph entities used", S["h"]),
                Paragraph(_esc(names), S["body"])]
    cits = getattr(ans, "citations", None) or ans.parsed.get("citations") or []
    if cits:
        out += [Paragraph("Citations (verified against the evidence)", S["h"]),
                Paragraph(_esc(", ".join(map(str, cits))), S["body"])]
    for s in getattr(ans, "sources", None) or []:
        out.append(Paragraph(f"[{_esc(s['id'])}] {_esc(s['statement'][:300])} "
                             f"<font color='#5B7290'>({_esc(s['dataset'])}, "
                             f"{_esc(s['table'])})</font>", S["ev"]))
    invalid = getattr(ans, "invalid_citations", None) or []
    if invalid:
        out.append(Paragraph("Citations NOT found in the evidence: " +
                             _esc(", ".join(map(str, invalid))), S["note"]))
    status = getattr(ans, "status", None)
    if status and status != "grounded":
        out.append(Paragraph(f"Grounding status: {_esc(status)}", S["note"]))

    if ans.retrieved_edge_ids:
        edges = [engine.edge_index[e] for e in ans.retrieved_edge_ids
                 if e in engine.edge_index]
        if edges:
            out.append(Paragraph("Evidence subgraph (blue = cited by the model)",
                                 S["h"]))
            png = subgraph_png(engine.G, edges, set(ans.cited_edge_ids))
            out.append(Image(BytesIO(png), width=6.4 * inch, height=4.1 * inch,
                             kind="proportional"))

    if ans.evidence:
        out.append(Paragraph("Evidence provided to the model", S["h"]))
        rows = [[Paragraph("<b>ID</b>", _STYLES["ev"]),
                 Paragraph("<b>Cited</b>", _STYLES["ev"]),
                 Paragraph("<b>Statement</b>", _STYLES["ev"])]]
        shown = sorted(ans.evidence, key=lambda t: not t[2])[:45]
        cited_rows = []
        for r, (uid, text, cited) in enumerate(shown, 1):
            body = text[len(uid) + 3:] if text.startswith(f"[{uid}] ") else text
            style = S["evc"] if cited else S["ev"]
            if cited:
                cited_rows.append(r)
            rows.append([Paragraph(_esc(uid), style),
                         _check_mark() if cited else "",
                         Paragraph(_esc(body[:320]), style)])
        tbl = Table(rows, colWidths=[0.85 * inch, 0.45 * inch, 5.2 * inch],
                    repeatRows=1)
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ] + [("BACKGROUND", (0, r), (-1, r), CITED_BG) for r in cited_rows]))
        out.append(tbl)
        if len(ans.evidence) > len(shown):
            out.append(Paragraph(
                f"({len(ans.evidence) - len(shown)} more evidence lines omitted; "
                "cited lines are shown first.)", S["note"]))

    method = ans.exposure_method if getattr(ans, "ai_relevant", False) else None
    if method:
        out.append(Paragraph("About the AI exposure score", S["h"]))
        out.append(Paragraph(_esc(method["meaning"]), S["body"]))
        out.append(Paragraph("<b>How it is calculated.</b> " + _esc(method["summary"]) +
                             _esc(f" Scores are taken from the {ai_exposure.SOURCE_NAME} "
                                  "(third-party, not part of O*NET) and matched to each "
                                  "O*NET occupation by SOC code and title."), S["body"]))
        out.append(Paragraph(_esc(method["formula"]), S["body"]))
        for r in ans.ai_exposure:
            x = r.get("exposure")
            if x:
                out.append(Paragraph(_esc(f"{r['title']}: {x['calculation']}"), S["note"]))
        missing = [r["title"] for r in ans.ai_exposure if not r.get("exposure")]
        if missing:
            out.append(Paragraph(_esc("Not covered by the index: " + ", ".join(missing)), S["note"]))
        for k, lim in enumerate(method["limitations"], 1):
            out.append(Paragraph(_esc(lim), S["note"], bulletText="-"))
        out.append(Paragraph("References", S["h"]))
        for k, ref in enumerate(method["references"], 1):
            out.append(Paragraph(_esc(f"[{k}] {ref['citation']} {ref['url']}"), S["note"]))

    u = ans.usage
    out.append(Paragraph(
        f"Context {ans.context_tokens} tokens | input {u.get('input_tokens')} / "
        f"output {u.get('output_tokens')} tokens | "
        f"latency {u.get('latency_s', 0):.1f}s", S["note"]))
    return out


def _occupation_rows(ans) -> list:
    """Table rows: the occupation asked about + the recommended ones."""
    S = _STYLES
    cards = {c["soc"]: c for c in getattr(ans, "occupations", None) or []}
    rows = []
    for r in getattr(ans, "ai_exposure", None) or []:
        x = r.get("exposure")
        c = cards.get(r["soc"], {})
        rows.append([Paragraph(_esc(r["title"]), S["ev"]),
                     Paragraph(_esc(r["soc"]), S["ev"]),
                     Paragraph(_esc(c.get("job_zone") or ""), S["ev"]),
                     Paragraph(_esc(f"#{c['related_rank']}" if c.get("related_rank") else
                                    r["role"]), S["ev"]),
                     Paragraph(_esc(f"{x['composite']:.2f}" if x else "n/a"), S["ev"]),
                     Paragraph(_esc(x["calculation"] if x else "not in index"), S["ev"])])
    if not rows:
        return []
    head = [Paragraph(f"<b>{h}</b>", S["ev"]) for h in
            ("Occupation", "O*NET-SOC", "Job Zone", "Related", "AI exposure", "Calculation")]
    return [head] + rows


def build_pdf(answers: list, engine, backend_desc: str, budget: int) -> bytes:
    S = _STYLES
    meta = (f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} | "
            f"LLM backend: {backend_desc} | evidence budget: {budget} tokens | "
            f"O*NET 30.3 knowledge graph: {engine.G.number_of_nodes():,} nodes / "
            f"{engine.G.number_of_edges():,} edges")

    from reportlab.platypus import HRFlowable

    story = [Spacer(1, 2.4 * inch),
             Paragraph(TITLE, S["title"]),
             Spacer(1, 0.12 * inch),
             Paragraph(SUBTITLE, S["subtitle"]),
             Spacer(1, 0.35 * inch),
             HRFlowable(width="34%", thickness=2.2, color=ACCENT,
                        hAlign="CENTER"),
             Spacer(1, 0.35 * inch),
             Paragraph("Graph-grounded question answering over the O*NET "
                       "occupational database. Every answer is retrieved by "
                       "traversing the knowledge graph and cited to "
                       "individual database records.", S["tagline"]),
             Spacer(1, 2.1 * inch),
             Paragraph(_esc(meta), S["meta"]),
             Spacer(1, 0.18 * inch),
             Paragraph("Data: O*NET 30.3 Database, U.S. Department of "
                       "Labor / ETA, used under CC BY 4.0. O*NET(R) is a "
                       "trademark of USDOL/ETA.", S["meta"]),
             PageBreak()]

    for i, ans in enumerate(answers, 1):
        story += _answer_flowables(i, ans, engine)
        if i < len(answers):
            story.append(PageBreak())

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title=f"{TITLE} — {SUBTITLE}",
                            author="O*NET Graph RAG")
    doc.build(story)
    return buf.getvalue()


# ------------------------------------------------------------ Markdown report
def _md_cites(text) -> str:
    """Citation ids as inline code so they stand out in any Markdown viewer."""
    return _CIT.sub(lambda m: " ".join(f"`{i.strip()}`" for i in m.group(1).split(",")),
                    str(text if text is not None else ""))


def _md_cell(text) -> str:
    return _md_cites(text).replace("|", "\\|").replace("\n", " ")


def _answer_markdown(i: int, ans, engine) -> list[str]:
    out = [f"## Q{i}. {ans.question.strip()}", ""]
    if ans.parsed is None:
        out += ["The model reply was not valid JSON:", "", "```", ans.raw[:1500], "```", ""]
        return out
    if ans.standalone_question and ans.standalone_question.strip() != ans.question.strip():
        out += [f"_Read with the conversation as: {ans.standalone_question.strip()}_", ""]
    if ans.headline:
        out += [f"> **{_md_cites(ans.headline)}**", ""]
    if ans.status == "ungrounded" or not ans.citations:
        out += ["> ⚠️ **Not based on O*NET evidence.** The model cited no retrieved record, "
                "so this answer (including any occupations it names) comes from the model "
                "itself and may not be relevant.", ""]
    elif ans.unsupported_occupations:
        out += ["> ⚠️ Listed without any cited evidence: " +
                ", ".join(ans.unsupported_occupations) + ".", ""]
    show_x = ans.exposure_in_evidence
    if ans.occupations:
        out += ["### Occupations", "",
                "| Occupation | O*NET-SOC | Job Zone | Related | "
                + ("AI exposure | " if show_x else "") + "Fit | Evidence cited |",
                "|---|---|---|---|" + ("---|" if show_x else "") + "---|---|"]
        for c in ans.occupations:
            x = c.get("ai_exposure")
            out.append("| " + " | ".join([
                _md_cell(c["title"]), c["soc"], str(c.get("job_zone") or "n/a"),
                f"#{c['related_rank']}" if c.get("related_rank") else "",
                *([f"{x['composite']:.2f}" if x else "not in index"] if show_x else []),
                _md_cell(c.get("fit") or ""),
                " ".join(f"`{s}`" for s in c.get("citations") or []) or "none"]) + " |")
        out.append("")
    a = ans.parsed.get("answer")
    out += ["### Answer", ""]
    if isinstance(a, list):
        out += [f"- {_md_cites(x)}" for x in a]
    elif isinstance(a, bool):
        out.append("**Yes**" if a else "**No**")
    elif a is not None:
        out.append(_md_cites(a).strip())
    out.append("")
    if ans.parsed.get("reasoning"):
        out += [f"_Reasoning: {_md_cites(ans.parsed['reasoning'])}_", ""]
    if ans.seeds:
        names = ", ".join(f"{engine.node_info(s)[0]} ({s})" for s in ans.seeds)
        out += [f"**Graph entities used:** {names}", ""]
    if ans.sources:
        out += [f"### Sources ({len(ans.sources)} records cited, verified against the evidence)", ""]
        out += [f"- `{s['id']}` {_md_cell(s['statement'][:300])} — _{s['dataset']}, {s['table']}_"
                for s in ans.sources] + [""]
    if ans.invalid_citations:
        out += ["Citations NOT found in the evidence: " + ", ".join(map(str, ans.invalid_citations)), ""]
    if ans.evidence:
        shown = sorted(ans.evidence, key=lambda t: not t[2])[:45]
        out += [f"### Evidence provided to the model ({len(ans.evidence)} lines; ✓ = cited)", "",
                "| ✓ | ID | Statement |", "|:-:|---|---|"]
        for uid, text, cited in shown:
            body = text[len(uid) + 3:] if text.startswith(f"[{uid}] ") else text
            out.append(f"| {'✓' if cited else ''} | `{uid}` | {_md_cell(body[:320])} |")
        if len(ans.evidence) > len(shown):
            out.append(f"| | | _{len(ans.evidence) - len(shown)} more lines omitted (cited lines first)_ |")
        out.append("")
    method = ans.exposure_method if ans.ai_relevant else None
    if method:
        out += ["### About the AI exposure score", "", method["meaning"], "",
                "**How it is calculated.** " + method["summary"], "",
                f"`{method['formula']}`", ""]
        for r in ans.ai_exposure:
            x = r.get("exposure")
            if x:
                out.append(f"- {r['title']}: `{x['calculation']}`")
        missing = [r["title"] for r in ans.ai_exposure if not r.get("exposure")]
        if missing:
            out.append("- Not covered by the index: " + ", ".join(missing))
        out += ["", "**References**", ""]
        out += [f"{k}. {ref['citation']} <{ref['url']}>"
                for k, ref in enumerate(method["references"], 1)] + [""]
    u = ans.usage
    out += [f"_Grounding: {ans.status} · model {ans.model or 'n/a'} · context "
            f"{ans.context_tokens} tokens · in {u.get('input_tokens')} / out "
            f"{u.get('output_tokens')} tokens · {u.get('latency_s', 0):.1f}s_", ""]
    return out


def build_markdown(answers: list, engine, backend_desc: str, budget: int) -> str:
    """The whole session as one Markdown document (same content as the PDF:
    every answer with its occupations, sources, evidence and - for AI
    questions - the score methodology)."""
    lines = [f"# {TITLE} — {SUBTITLE}", "",
             f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} · LLM: {backend_desc} · "
             f"evidence budget {budget} tokens · O*NET 30.3 knowledge graph: "
             f"{engine.G.number_of_nodes():,} nodes / {engine.G.number_of_edges():,} edges", "",
             "Graph-grounded question answering over the O*NET occupational database. Every "
             "answer is retrieved by traversing the knowledge graph and cited to individual "
             "database records; each citation is verified against the evidence the model saw.",
             "",
             "Data: O*NET 30.3 Database, U.S. Department of Labor / ETA (CC BY 4.0). O*NET® is a "
             "trademark of USDOL/ETA; this tool is not reviewed or endorsed by USDOL/ETA. "
             "AI exposure: " + ai_exposure.SOURCE_CITATION +
             " (third-party index, not part of O*NET).", "", "---", ""]
    for i, ans in enumerate(answers, 1):
        lines += _answer_markdown(i, ans, engine)
        lines += ["---", ""]
    return "\n".join(lines)
