"""PDF report of chat results, with a clean title page.

build_pdf(answers, ...) -> bytes for st.download_button. Evidence subgraphs
are rendered as static matplotlib images (pyvis is JS-only, so the PDF gets
its own drawing of the same edges).
"""
from __future__ import annotations

import datetime
from io import BytesIO

import bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from graph_view import LABEL_COLORS

TITLE = "O*NET Graph RAG"
SUBTITLE = "Results Report"
NAVY = colors.HexColor("#10304F")
ACCENT = colors.HexColor("#1D4ED8")

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
                          textColor=colors.HexColor("#b02020")),
    "note": ParagraphStyle("n", parent=_SS["Normal"], fontSize=8.5,
                           leading=11, textColor=colors.grey),
}


def _esc(s) -> str:
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.encode("latin-1", "replace").decode("latin-1")


def subgraph_png(G, edges, cited_edge_ids: set[int], max_edges: int = 60) -> bytes:
    """Static drawing of the evidence subgraph; cited edges red and thick."""
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
        ax.plot([x1, x2], [y1, y2], color="#c03030" if cited else "#c9c9c9",
                lw=1.8 if cited else 0.7, alpha=0.95 if cited else 0.65,
                zorder=2 if cited else 1)
    handles = [plt.Line2D([], [], marker="o", ls="", color=LABEL_COLORS.get(l, "#999"),
                          label=l, markersize=6) for l in labels_present]
    handles.append(plt.Line2D([], [], color="#c03030", lw=2, label="cited edge"))
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

    a = ans.parsed.get("answer")
    out.append(Paragraph("Answer", S["h"]))
    if isinstance(a, list):
        for x in a:
            out.append(Paragraph("• " + _esc(x), S["answer"]))
    else:
        out.append(Paragraph(_esc(a), S["answer"]))
    if ans.parsed.get("reasoning"):
        out += [Paragraph("Reasoning", S["h"]),
                Paragraph(_esc(ans.parsed["reasoning"]), S["body"])]
    if ans.seeds:
        names = ", ".join(f"{engine.node_info(s)[0]} ({s})" for s in ans.seeds)
        out += [Paragraph("Graph entities used", S["h"]),
                Paragraph(_esc(names), S["body"])]
    if ans.parsed.get("citations"):
        out += [Paragraph("Citations", S["h"]),
                Paragraph(_esc(", ".join(map(str, ans.parsed["citations"]))),
                          S["body"])]

    if ans.retrieved_edge_ids:
        edges = [engine.edge_index[e] for e in ans.retrieved_edge_ids
                 if e in engine.edge_index]
        if edges:
            out.append(Paragraph("Evidence subgraph (red = cited by the model)",
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
        for uid, text, cited in shown:
            body = text[len(uid) + 3:] if text.startswith(f"[{uid}] ") else text
            style = S["evc"] if cited else S["ev"]
            rows.append([Paragraph(_esc(uid), style),
                         Paragraph("X" if cited else "", style),
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
        ]))
        out.append(tbl)
        if len(ans.evidence) > len(shown):
            out.append(Paragraph(
                f"({len(ans.evidence) - len(shown)} more evidence lines omitted; "
                "cited lines are shown first.)", S["note"]))

    u = ans.usage
    out.append(Paragraph(
        f"Context {ans.context_tokens} tokens | input {u.get('input_tokens')} / "
        f"output {u.get('output_tokens')} tokens | "
        f"latency {u.get('latency_s', 0):.1f}s", S["note"]))
    return out


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
