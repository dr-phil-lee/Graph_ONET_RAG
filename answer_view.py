"""Structured, graphical rendering of one chatbot Answer (Streamlit).

Layout of an answer:
  grounding notice (only when the evidence did not cover the question)
  -> headline banner -> key points -> occupation cards (Job Zone, O*NET
  relatedness rank, AI-exposure gauge, top transferable skills, citations)
  -> tabs: Answer · Focus graph · Evidence graph · Sources · Evidence · Charts
  -> what the AI exposure score means and how it was computed (only when
     the question was about AI)
Everything except the model's own wording (headline, key points, "fit",
answer) is read from the graph or the AI Exposure Index, never from the
model. Model text is HTML-escaped before it is placed in HTML.
"""
from __future__ import annotations

import html as _html
import re

import altair as alt
import pandas as pd
import streamlit as st

import ai_exposure
import graph_view

CSS = """
<style>
.ans-headline {background: linear-gradient(120deg, rgba(37,99,235,.10), rgba(56,189,248,.06));
  border: 1px solid rgba(37,99,235,.22); border-left: 6px solid #2563EB;
  border-radius: 14px; padding: .85rem 1.1rem; margin: .2rem 0 .7rem 0;}
.ans-headline .lbl {font-size: .7rem; font-weight: 700; letter-spacing: .9px;
  text-transform: uppercase; color: #2563EB;}
.ans-headline .txt {font-size: 1.12rem; font-weight: 650; line-height: 1.45;
  color: inherit; margin-top: .15rem;}
.ans-interp {font-size: .8rem; color: #4B5563; margin: -.35rem 0 .6rem .2rem;}
.note {border: 1px solid #D1D5DB; border-left: 5px solid #6B7280; border-radius: 10px;
  background: #F9FAFB; padding: .5rem .8rem .45rem; margin: .1rem 0 .7rem; font-size: .84rem;
  line-height: 1.45; color: #1F2937;}
.note .lbl {font-size: .68rem; font-weight: 800; letter-spacing: .8px; text-transform: uppercase;
  color: #374151; margin-bottom: .2rem;}
.note ul {margin: 0; padding-left: 1.1rem;}
.note li {margin: .08rem 0;}
.note li.cur {background: #EFF4FF; border-radius: 6px; padding: .05rem .35rem; margin-left: -.35rem;}
.note b {color: #111827;}
.cite {display:inline-block; font: 600 .68rem/1.4 ui-monospace, Consolas, monospace;
  color:#1D4ED8; background: rgba(37,99,235,.09); border: 1px solid rgba(37,99,235,.25);
  border-radius: 6px; padding: 0 .35rem; margin: 0 .1rem; vertical-align: 1px;}
.cite.x {color:#9A3412; background: rgba(234,88,12,.08); border-color: rgba(234,88,12,.3);}
.occ-card {border: 1px solid rgba(120,130,150,.28); border-radius: 14px;
  padding: .8rem .9rem .7rem; height: 100%; background: rgba(255,255,255,.02);
  box-shadow: 0 2px 10px rgba(15,23,42,.05); margin-bottom: .6rem;}
.occ-card .top {display:flex; justify-content:space-between; align-items:center; gap:.4rem;}
.occ-card .rank {font-size:.68rem; font-weight:700; color:#fff; background:#334155;
  border-radius:999px; padding:.08rem .5rem; white-space:nowrap;}
.occ-card .soc {font: .7rem ui-monospace, Consolas, monospace; color:#4B5563;}
.occ-card .ttl {font-weight:750; font-size:1.0rem; line-height:1.3; margin:.35rem 0 .2rem;}
.occ-card .desc {font-size:.8rem; color:#374151; line-height:1.45; margin-bottom:.45rem;}
.occ-card .nocite {font-size:.68rem; font-weight:700; color:#92400E; background:#FEF3C7;
  border:1px solid #F59E0B; border-radius:999px; padding:.08rem .5rem; white-space:nowrap;}
.occ-card .fit {font-size:.86rem; line-height:1.45; margin:.35rem 0 .45rem;}
.occ-card .row {display:flex; align-items:center; gap:.45rem; font-size:.78rem;
  margin:.2rem 0; color:#1F2937;}
.occ-card .k {flex:0 0 5.3rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.4px; font-size:.64rem; color:#475569;}
.jz {display:inline-flex; gap:3px;}
.jz i {width:11px; height:11px; border-radius:3px; background:#E2E8F0; display:inline-block;}
.jz i.on {background:#2563EB;}
.xbar {flex:1; display:flex; align-items:center; gap:.4rem; min-width:7rem;}
.xbar .track {position:relative; flex:1; height:12px; border-radius:6px; background:#E5E7EB;
  overflow:hidden;}
.xbar .fill {position:absolute; left:0; top:0; height:100%; border-radius:6px;}
.xbar .tick {position:absolute; top:0; width:1px; height:100%; background:rgba(17,24,39,.18);}
.xbar .lo, .xbar .hi {font-size:.6rem; color:#6B7280; font-weight:600;}
.pill {display:inline-block; font-size:.68rem; font-weight:700; color:#fff;
  border-radius:999px; padding:.05rem .5rem; white-space:nowrap;}
.chip {display:inline-block; font-size:.72rem; background:rgba(100,116,139,.12);
  border-radius:6px; padding:.05rem .4rem; margin:.1rem .15rem 0 0; color:#1F2937;}
.sec-title {font-size:.78rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase;
  color:#475569; margin:.7rem 0 .35rem;}
.src-row {font-size:.88rem; line-height:1.45; padding:.3rem 0;
  border-bottom:1px solid rgba(120,130,150,.15);}
.src-meta {color:#475569; font-size:.76rem;}
.method {border:1px solid rgba(37,99,235,.25); border-left:6px solid #2563EB;
  border-radius:14px; padding:.75rem 1rem .6rem; margin:.6rem 0 .4rem;
  background: rgba(37,99,235,.04);}
.method .lbl {font-size:.7rem; font-weight:700; letter-spacing:.9px;
  text-transform:uppercase; color:#1D4ED8;}
.method .meaning {font-size:.95rem; line-height:1.55; margin:.25rem 0 .6rem; font-weight:500;}
.method .sum {font-size:.86rem; line-height:1.5; margin:.2rem 0 .45rem; color:#1F2937;}
.method .formula {font: 600 .86rem/1.5 ui-monospace, Consolas, monospace;
  background: rgba(15,23,42,.06); border-radius:8px; padding:.3rem .6rem;
  display:inline-block; margin-bottom:.5rem;}
.method .dims {display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.5rem;}
.method .dim {flex:1 1 12rem; border:1px solid rgba(120,130,150,.25); border-radius:10px;
  padding:.4rem .55rem; font-size:.76rem; line-height:1.4; background:rgba(255,255,255,.35);}
.method .dim b {font-size:.8rem;}
.method .dir {font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.4px;}
.method .calc {font-size:.8rem; line-height:1.55; margin:.1rem 0 .45rem;}
.method .calc code {font-size:.76rem;}
.method .refs {font-size:.76rem; line-height:1.5; color:#374151; margin:0; padding-left:1.1rem;}
.method .refs a {color:#1D4ED8;}
</style>
"""

_CIT = re.compile(r"\[\s*((?:E\d+|[NXD]-[0-9A-Za-z.\-]+)(?:\s*,\s*(?:E\d+|[NXD]-[0-9A-Za-z.\-]+))*)\s*\]")


def rich(text: str) -> str:
    """Escape model text, then style [E123] / [N-...] / [X-...] ids as chips
    and **bold** as bold. Safe to pass to unsafe_allow_html."""
    t = _html.escape(str(text or ""))

    def chips(m: re.Match) -> str:
        ids = [i.strip() for i in m.group(1).split(",")]
        return "".join(f"<span class='cite{' x' if i.startswith('X-') else ''}'>{i}</span>"
                       for i in ids)

    t = _CIT.sub(chips, t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t


def md_with_chips(text: str) -> str:
    """For st.markdown answers: keep markdown, render ids as code spans."""
    return _CIT.sub(lambda m: " ".join(f"`{i.strip()}`" for i in m.group(1).split(",")),
                    str(text or ""))


# ------------------------------------------------------------------ pieces
_DIM_ORDER = [("Digital", "raises exposure", "#C2622F"),
              ("Human contact", "lowers exposure", "#6E3A55"),
              ("Physical", "lowers exposure", "#3B6E8F")]


def method_html(ans=None) -> str:
    """What the AI exposure score means, then how it was computed: source,
    data, formula, each occupation's own calculation, and the references.
    Text is fixed (from ai_exposure.METHOD_* / MEANING), not model output.
    With no answer the general explanation is returned (no calculations)."""
    if ans is None:
        m = ai_exposure.methodology()
    else:
        m = ans.exposure_method
        if not m:
            return ""
    esc = _html.escape
    dims = []
    for name, effect, col in _DIM_ORDER:
        vs = [v["variable"] for v in m["variables"] if v["dimension"] == name]
        dims.append(f"<div class='dim'><b style='color:{col}'>{esc(name)}</b> "
                    f"<span class='dir' style='color:{col}'>{esc(effect)}</span><br>"
                    + "; ".join(esc(v) for v in vs) + "</div>")
    calc = []
    for r in (ans.ai_exposure if ans is not None else []):
        x = r.get("exposure")
        if x:
            calc.append(f"<b>{esc(r['title'])}</b>: <code>{esc(x['calculation'])}</code>")
    missing = [r["title"] for r in (ans.ai_exposure if ans is not None else [])
               if not r.get("exposure")]
    if missing:
        calc.append("Not covered by the index (no score): " + esc(", ".join(missing)))
    if calc:
        calc.insert(0, "<span class='lbl'>Occupations in your latest answer about AI</span>")
    refs = "".join(f"<li>{esc(r['citation'])} <a href='{esc(r['url'])}' target='_blank'>"
                   f"{esc(r['url'])}</a></li>" for r in m["references"])
    return f"""
<div class='method'>
  <div class='lbl'>About the AI exposure score</div>
  <div class='meaning'>{esc(m['meaning'])}</div>
  <div class='lbl'>How the AI exposure scores are calculated</div>
  <div class='sum'>{esc(m['summary'])} Scores are taken from the {esc(ai_exposure.SOURCE_NAME)}
  (third-party, not part of O*NET) and matched to each O*NET occupation by SOC code and title.</div>
  <div class='formula'>{esc(m['formula'])}</div>
  <div class='dims'>{''.join(dims)}</div>
  <div class='calc'>{'<br>'.join(calc)}</div>
  <div class='lbl' style='margin-bottom:.15rem'>References</div>
  <ol class='refs'>{refs}</ol>
</div>"""


def method_details(ans=None) -> None:
    """The nine O*NET variables (table) and the data-join note."""
    m = ai_exposure.methodology() if ans is None else ans.exposure_method
    if not m:
        return
    st.dataframe(pd.DataFrame(m["variables"]).rename(columns={
        "variable": "O*NET variable", "onet_element": "O*NET element",
        "scale": "O*NET section / scale", "dimension": "Dimension",
        "direction": "Direction"}), hide_index=True, width="stretch")

    def md(s: str) -> str:            # "O*NET" must not start italics
        return s.replace("*", "\\*")

    st.caption(md(m["join"] + " The index uses O*NET 31.0; the chatbot graph is O*NET 30.3."))


METHOD_PROMPT = "Where does the AI exposure score come from?"


def exposure_panel(ans=None) -> None:
    """Body of the "Where does the AI exposure score come from?" panel in the
    info bar (kept apart from the chat): what the score means, how it is
    calculated, the full methodology and references. When the latest answer
    about AI is passed, its occupations' own calculations are included."""
    if ans is not None and not (getattr(ans, "ai_relevant", False) and ans.exposure_method):
        ans = None
    st.markdown(method_html(ans), unsafe_allow_html=True)
    method_details(ans)


_NOT_COVERED = re.compile(r"^\s*(insufficient evidence|the evidence does not cover|"
                          r"no evidence)", re.I)


def grounding_notice(ans) -> None:
    """One clear statement, before anything else, when the answer is not
    backed by the retrieved O*NET records - so the reader never has to work
    out whether a list of occupations came from the data or from the model."""
    jobs = [e.name for e in ans.entities if e.label == "Job"]
    about = f" about {', '.join(jobs[:3])}" if jobs else ""
    if ans.status == "ungrounded" or (not ans.citations and ans.parsed is not None):
        st.error("**This answer is not based on O*NET evidence.** The model cited no "
                 f"retrieved record{about}, so anything it says - including any "
                 "occupations it names - comes from the model itself and may not be "
                 "relevant to your question. Try naming the occupation exactly as "
                 "O*NET does, or confirm one of the suggested matches.", icon="⚠️")
        return
    if _NOT_COVERED.match(ans.headline or ""):
        st.warning("**The retrieved O*NET records do not cover what you asked.** The "
                   "answer below says what the evidence does and does not contain; no "
                   "occupation is recommended without a cited record.", icon="⚠️")
    unsupported = ans.unsupported_occupations
    if unsupported:
        st.warning("**Not supported by the evidence:** " + ", ".join(unsupported) +
                   ". The model listed these occupations without citing any O*NET "
                   "record for them (marked on the cards), so treat them as the "
                   "model's own suggestion, not as a data-based result.", icon="⚠️")


def headline(ans) -> None:
    if not ans.headline:
        return
    st.markdown(f"<div class='ans-headline'><div class='lbl'>Answer</div>"
                f"<div class='txt'>{rich(ans.headline)}</div></div>",
                unsafe_allow_html=True)
    if ans.standalone_question and ans.standalone_question.strip() != ans.question.strip():
        st.markdown(f"<div class='ans-interp'>🧠 Read with the conversation as: "
                    f"<i>{_html.escape(ans.standalone_question)}</i></div>",
                    unsafe_allow_html=True)


def _jz_html(jz) -> str:
    if jz is None:
        return "<span class='chip'>n/a</span>"
    return ("<span class='jz'>" + "".join(f"<i class='{'on' if i <= jz else ''}'></i>"
                                         for i in range(1, 6))
            + f"</span> <b>{jz}</b>/5")


def _gauge_html(x: dict | None) -> str:
    """Horizontal bar graph of the AI exposure score on a 0-1 track (quarter
    ticks), filled in the score's colour, with the value as a pill."""
    if not x:
        return "<span class='chip'>not in AI Exposure Index</span>"
    pos = max(0.0, min(1.0, float(x["composite"]))) * 100
    ticks = "".join(f"<span class='tick' style='left:{t}%'></span>" for t in (25, 50, 75))
    return (f"<div class='xbar' title='AI exposure {x['composite']:.2f} on a 0-1 scale'>"
            f"<span class='lo'>0</span><div class='track'>{ticks}"
            f"<div class='fill' style='width:{pos:.1f}%;background:{x['color']}'></div></div>"
            f"<span class='hi'>1</span>"
            f"<span class='pill' style='background:{x['color']}'>{x['composite']:.2f}</span></div>")


def occupation_card_html(c: dict, current: dict | None = None,
                         show_exposure: bool = True) -> str:
    """One occupation card. The AI exposure bar graph is always drawn under
    Job Zone; `show_exposure` only adds the difference to the current job
    (AI / career questions)."""
    rank = (f"<span class='rank'>#{c['related_rank']} related</span>"
            if c.get("related_rank") else "")
    if c.get("source") == "model" and not c.get("citations"):
        rank += "<span class='nocite'>no evidence cited</span>"
    x = c.get("ai_exposure")
    delta = ""
    if show_exposure and x and current and current.get("composite") is not None:
        d = x["composite"] - current["composite"]
        arrow = "▼" if d < 0 else "▲"
        col = "#2E7D5B" if d < 0 else "#B23A3A"
        delta = (f"<span style='color:{col};font-weight:700;font-size:.72rem'>"
                 f"{arrow} {abs(d):.2f} vs. current</span>")
    skills = "".join(f"<span class='chip'>{_html.escape(s['name'])}"
                     f"{' · ' + format(s['importance'], '.2f') if s.get('importance') else ''}</span>"
                     for s in c.get("top_skills") or [])
    cites = "".join(f"<span class='cite{' x' if i.startswith('X-') else ''}'>{_html.escape(i)}</span>"
                    for i in c.get("citations") or [])
    fit = f"<div class='fit'>{rich(c['fit'])}</div>" if c.get("fit") else ""
    return f"""
<div class='occ-card'>
  <div class='top'><span class='soc'>{_html.escape(c['soc'])}</span>{rank}</div>
  <div class='ttl'>{_html.escape(c['title'])}</div>
  <div class='desc'>{_html.escape(c.get('description') or '')}</div>
  {fit}
  <div class='row'><span class='k'>Job Zone</span>{_jz_html(c.get('job_zone'))}</div>
  <div class='row'><span class='k'>AI exposure</span>{_gauge_html(x)}</div>
  {f"<div class='row'><span class='k'></span>{delta}</div>" if delta else ""}
  {f"<div class='row' style='flex-wrap:wrap'><span class='k'>Top skills</span><span>{skills}</span></div>" if skills else ""}
  {f"<div class='row' style='flex-wrap:wrap'><span class='k'>Sources</span><span>{cites}</span></div>" if cites else ""}
</div>"""


def current_job_strip(ans) -> dict | None:
    """Compact profile of the occupation asked about (one current job)."""
    cur = [r for r in ans.ai_exposure if r["role"] == "current"]
    if len(cur) != 1:
        return None
    return cur[0]


def occupation_cards(ans, engine) -> None:
    show_x = ans.exposure_in_evidence      # scores only for AI / career questions
    cur = current_job_strip(ans)
    if cur is not None:
        card = engine.occupation_card(cur["soc"])
        card.update(fit="", citations=[c for c in ans.citations
                                       if c in (f"N-{cur['soc']}", f"X-{cur['soc']}")])
        st.markdown("<div class='sec-title'>Your occupation</div>", unsafe_allow_html=True)
        st.markdown(occupation_card_html(card, show_exposure=show_x), unsafe_allow_html=True)
    if not ans.occupations:
        return
    label = "Recommended occupations" if cur is not None else "Occupations"
    st.markdown(f"<div class='sec-title'>{label} ({len(ans.occupations)})</div>",
                unsafe_allow_html=True)
    cur_x = (cur or {}).get("exposure") if show_x else None
    per_row = 2 if len(ans.occupations) <= 4 else 3
    cols = st.columns(per_row)
    for i, c in enumerate(ans.occupations):
        with cols[i % per_row]:
            st.markdown(occupation_card_html(c, cur_x, show_exposure=show_x),
                        unsafe_allow_html=True)


# ------------------------------------------------------------------ charts
def _score_scale():
    """Continuous green-red scale for AI exposure scores (no named bands)."""
    return alt.Scale(domain=[s for s, _ in ai_exposure.COLOR_STOPS],
                     range=[c for _, c in ai_exposure.COLOR_STOPS], clamp=True)


def exposure_frame(ans) -> pd.DataFrame:
    rows = []
    for r in ans.ai_exposure:
        x = r.get("exposure")
        if not x:
            continue
        tag = {"current": " (current)", "recommended": "", "discussed": ""}[r["role"]]
        rows.append({"occupation": r["title"] + tag, "role": r["role"],
                     "composite": x["composite"],
                     "digital": x["digital"], "human contact": x["human_contact"],
                     "physical": x["physical"], "employment": x["employment"],
                     "soc": r["soc"]})
    return pd.DataFrame(rows)


def exposure_chart(df: pd.DataFrame) -> alt.Chart:
    order = df.sort_values("composite")["occupation"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("occupation:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=alt.X("composite:Q", scale=alt.Scale(domain=[0, 1]),
                title="Composite AI exposure (0 = protected, 1 = exposed)"),
        color=alt.Color("composite:Q", scale=_score_scale(), legend=None),
        stroke=alt.condition("datum.role == 'current'", alt.value("#111827"), alt.value(None)),
        strokeWidth=alt.condition("datum.role == 'current'", alt.value(2), alt.value(0)),
        tooltip=["occupation", "soc", alt.Tooltip("composite:Q", format=".4f"),
                 alt.Tooltip("digital:Q", format=".2f"),
                 alt.Tooltip("human contact:Q", format=".2f"),
                 alt.Tooltip("physical:Q", format=".2f"),
                 alt.Tooltip("employment:Q", format=",")])
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        x="composite:Q", text=alt.Text("composite:Q", format=".2f"))
    # step sizing: a fixed pixel height collapses the band scale inside
    # st.altair_chart
    return (bars + labels).properties(height=alt.Step(34))


def dimension_chart(df: pd.DataFrame) -> alt.Chart:
    long = df.melt(id_vars=["occupation"], value_vars=["digital", "human contact", "physical"],
                   var_name="dimension", value_name="value")
    return alt.Chart(long).mark_bar(cornerRadiusEnd=3, height=9).encode(
        y=alt.Y("occupation:N", title=None, axis=alt.Axis(labelLimit=260),
                sort=df.sort_values("composite")["occupation"].tolist()),
        yOffset="dimension:N",
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Intensity (0-1)"),
        color=alt.Color("dimension:N", scale=alt.Scale(
            domain=["digital", "human contact", "physical"],
            range=["#C2622F", "#6E3A55", "#3B6E8F"]),
            legend=alt.Legend(orient="bottom", title="Dimension (digital raises, the others "
                                                      "protect)")),
        tooltip=["occupation", "dimension", alt.Tooltip("value:Q", format=".4f")],
    ).properties(height=alt.Step(13))


def pivot_map(ans, engine) -> alt.Chart | None:
    """Job Zone (preparation) vs AI exposure for the occupations in focus."""
    rows = []
    jobs = {r["soc"]: r for r in ans.ai_exposure}
    for soc, r in jobs.items():
        x = r.get("exposure")
        jz = None
        for c in ans.occupations:
            if c["soc"] == soc:
                jz = c.get("job_zone")
        if jz is None:
            jz = _job_zone(engine, soc)
        if x and jz is not None:
            rows.append({"occupation": r["title"], "role": r["role"], "AI exposure": x["composite"],
                         "Job Zone": jz})
    if len(rows) < 2:
        return None
    df = _place_labels(pd.DataFrame(rows))
    x_enc = alt.X("AI exposure:Q", scale=alt.Scale(domain=[0.2, 0.9]),
                  title="Composite AI exposure →")
    y_enc = alt.Y("Job Zone:Q", scale=alt.Scale(domain=[0.4, 5.7]),
                  title="Job Zone (preparation needed) →", axis=alt.Axis(values=[1, 2, 3, 4, 5]))
    pts = alt.Chart(df).mark_point(filled=True, size=220, opacity=.9).encode(
        x=x_enc, y=y_enc,
        color=alt.Color("AI exposure:Q", scale=_score_scale(), legend=None),
        shape=alt.Shape("role:N", scale=alt.Scale(domain=["current", "recommended", "discussed"],
                                                  range=["diamond", "circle", "square"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["occupation", "role", "Job Zone", alt.Tooltip("AI exposure:Q", format=".4f")])
    txt = alt.Chart(df).mark_text(align="left", dx=10, fontSize=11).encode(
        x=x_enc, y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0.4, 5.7])), text="label:N")
    leaders = alt.Chart(df).mark_rule(color="#9CA3AF", strokeWidth=1).encode(
        x=x_enc, y=y_enc, y2="label_y:Q")
    return (leaders + pts + txt).properties(height=340)


# conservative: a 0.7-wide x range on a ~500px plot; wider charts only gain room
_PX_PER_X = 650         # ≈ pixels per 1.0 of AI exposure
_PX_PER_CHAR = 6.6      # ≈ width of one character at 11px
_ZONE_PX = 56           # ≈ pixels per Job Zone step (340px / 6 zones)


def _place_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Occupations in the same Job Zone sit on one horizontal line, so
    their labels would run into each other. Labels are placed greedily:
    within a zone (left to right) each label takes the first stacking level
    whose previous label has already ended; level 0 is just above the point,
    level 1 just below, then further above / below. `label_y` is the label's
    y in data units, drawn with a thin leader to the point."""
    df = df.sort_values(["Job Zone", "AI exposure"]).reset_index(drop=True)
    df["label"] = df["occupation"].map(lambda s: s if len(s) <= 36 else s[:35] + "…")
    label_y = []
    for _, grp in df.groupby("Job Zone", sort=False):
        ends: list[float] = []            # per level: pixel x where its last label ends
        for x, lab, zone in zip(grp["AI exposure"], grp["label"], grp["Job Zone"]):
            px = float(x) * _PX_PER_X
            for lvl, end in enumerate(ends):
                if end < px - 4:
                    break
            else:
                lvl = len(ends)
                ends.append(0.0)
            ends[lvl] = px + 12 + _PX_PER_CHAR * len(lab)
            step = 14 / _ZONE_PX                              # ≈ one text line, in zone units
            if lvl % 2 == 0:
                label_y.append(float(zone) + 0.16 + step * (lvl // 2))
            else:
                label_y.append(float(zone) - 0.26 - step * (lvl // 2))
    df["label_y"] = label_y
    return df


def _job_zone(engine, soc: str):
    if soc not in engine.G:
        return None
    jz = engine.G.nodes[soc].get("job_zone")
    return int(jz) if jz is not None and jz == jz else None


def ratings_frame(ans, engine) -> pd.DataFrame:
    """Importance ratings among the retrieved evidence (cited marked)."""
    cited = set(ans.cited_edge_ids)
    rows = []
    for eid in ans.retrieved_edge_ids:
        e = engine.edge_index.get(eid)
        if not e or e[3].get("type") != "RATED":
            continue
        u, v, k, d = e
        if d.get("importance") is None:
            continue
        rows.append({"occupation": engine.G.nodes[u].get("name", u),
                     "descriptor": engine.G.nodes[v].get("name", v),
                     "kind": engine.G.nodes[v].get("label", ""),
                     "importance": float(d["importance"]),
                     "level": float(d["level"]) if d.get("level") is not None else None,
                     "cited": "cited" if int(k) in cited else "retrieved",
                     "id": f"E{k}"})
    return pd.DataFrame(rows)


def ratings_chart(df: pd.DataFrame) -> alt.Chart:
    df = df.sort_values("importance", ascending=False).head(24)
    order = df.groupby("descriptor")["importance"].max().sort_values(ascending=False).index.tolist()
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("descriptor:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset="occupation:N",          # one bar per occupation, never stacked
        x=alt.X("importance:Q", scale=alt.Scale(domain=[0, 5]), stack=None,
                title="O*NET importance (1-5)"),
        color=alt.Color("occupation:N", legend=alt.Legend(orient="bottom", title=None,
                                                          labelLimit=260)),
        opacity=alt.condition("datum.cited == 'cited'", alt.value(1.0), alt.value(.45)),
        tooltip=["id", "occupation", "descriptor", "kind",
                 alt.Tooltip("importance:Q", format=".2f"),
                 alt.Tooltip("level:Q", format=".2f"), "cited"],
    ).properties(height=alt.Step(11))


def charts(ans, engine) -> None:
    shown = False
    df = exposure_frame(ans) if ans.exposure_in_evidence else pd.DataFrame()
    if not df.empty:
        shown = True
        st.markdown("**AI exposure of the occupations in this answer**")
        st.altair_chart(exposure_chart(df), width="stretch")
        st.caption(f"Source: {ai_exposure.SOURCE_CITATION}. {ai_exposure.SOURCE_NOTE} "
                   "Bar color follows the score (green = lower, red = higher); "
                   "a dark outline marks your current occupation.")
        st.markdown("**What drives the score**")
        st.altair_chart(dimension_chart(df), width="stretch")
        pm = pivot_map(ans, engine)
        if pm is not None:
            st.markdown("**Pivot map: preparation vs. AI exposure** "
                        "(lower-left = less preparation and less exposed)")
            st.altair_chart(pm, width="stretch")
    missing = [r["title"] for r in ans.ai_exposure if not r.get("exposure")]
    if missing and ans.exposure_in_evidence:
        st.caption("Not covered by the AI Exposure Index: " + ", ".join(missing) + ".")
    rf = ratings_frame(ans, engine)
    if not rf.empty:
        shown = True
        st.markdown("**O*NET ratings in the evidence** (solid = cited by the answer)")
        st.altair_chart(ratings_chart(rf), width="stretch")
    if not shown:
        st.info("No chartable ratings or AI-exposure scores in this answer's evidence.")


# ------------------------------------------------------------------ graphs
def exposure_map(engine, nodes) -> dict[str, dict]:
    out = {}
    for n in nodes:
        if n in engine.G and engine.G.nodes[n].get("label") == "Job":
            x = engine.exposure_for(n)
            if x:
                out[n] = x
    return out


def focus_graph_html(ans, engine) -> str | None:
    edges = [engine.edge_index[e] for e in ans.focus_edge_ids if e in engine.edge_index]
    if not edges and not ans.focus_node_ids:
        return None
    nodes = set(ans.focus_node_ids) | {n for u, v, _, _ in edges for n in (u, v)}
    seeds = {e.node_id for e in ans.entities if e.label == "Job"}
    return graph_view.render_edges_html(
        engine.G, edges, engine.anchors, cited_edge_ids=set(ans.cited_edge_ids),
        height=520, extra_nodes=ans.focus_node_ids, focus_nodes=seeds,
        exposure=exposure_map(engine, nodes),
        legend=graph_view.exposure_legend() + [("#d62728", "— cited record"),
                                               ("#c9c9c9", "- - connecting O*NET relation")])


def evidence_graph_html(ans, engine) -> str | None:
    edges = [engine.edge_index[e] for e in ans.retrieved_edge_ids if e in engine.edge_index]
    if not edges:
        return None
    return graph_view.render_edges_html(engine.G, edges, engine.anchors,
                                        cited_edge_ids=set(ans.cited_edge_ids), height=560)


def sources(ans) -> None:
    if not ans.sources:
        st.info("The answer cited no evidence.")
        return
    groups: dict[str, list[dict]] = {}
    for s in ans.sources:
        groups.setdefault(s["table"], []).append(s)
    summary = pd.DataFrame([{"source table": t, "records cited": len(v),
                             "dataset": v[0]["dataset"]} for t, v in groups.items()])
    st.dataframe(summary, hide_index=True, width="stretch")
    for s in ans.sources:
        url = f" · <a href='{s['url']}' target='_blank'>source</a>" if s.get("url") else ""
        st.markdown(f"<div class='src-row'><span class='cite{' x' if s['id'].startswith('X-') else ''}'>"
                    f"{_html.escape(s['id'])}</span> {_html.escape(s['statement'])}<br>"
                    f"<span class='src-meta'>{_html.escape(s['dataset'])} · "
                    f"{_html.escape(s['table'])}{url}</span></div>", unsafe_allow_html=True)


def evidence_table(ans) -> None:
    rows = sorted(ans.evidence, key=lambda t: not t[2])
    df = pd.DataFrame(
        [{"Cited": "✅" if cited else "", "ID": uid,
          "Statement": (text[len(uid) + 3:] if text.startswith(f"[{uid}] ") else text)}
         for uid, text, cited in rows])
    st.dataframe(df, hide_index=True, width="stretch", height=380)


VIEW_NOTES = [
    ("📝 Answer", "the written answer: headline, text with citation chips, and a card "
                  "per occupation (Job Zone, AI exposure bar, top skills, sources)."),
    ("🎯 Focus graph", "the retrieved O*NET records that are directly linked to the "
                       "occupation(s) you asked about (★). It is the same whichever "
                       "model answers; red = records this model cited, grey dashed = "
                       "retrieved but not cited."),
    ("🕸️ Evidence graph", "everything retrieved for this question - the whole evidence "
                          "the model was given, including related occupations and their "
                          "own records, so it is larger than the Focus graph."),
    ("📚 Sources", "the O*NET records the answer actually cites, grouped by the O*NET "
                   "table each comes from."),
    ("🔎 Evidence", "every evidence line handed to the model, as text; ✓ marks the "
                    "lines the answer cites."),
    ("📊 Charts", "the O*NET ratings in the evidence and, for questions about AI, the "
                  "AI exposure scores of the occupations in this answer."),
]


def views_note_html(current: str | None = None) -> str:
    """The 'Note' box shown under the six view buttons: one line per view;
    the open view is highlighted."""
    items = "".join(
        f"<li class='{'cur' if name == current else ''}'><b>{_html.escape(name)}</b> — "
        f"{_html.escape(text)}</li>" for name, text in VIEW_NOTES)
    return (f"<div class='note'><div class='lbl'>Note — what each view shows</div>"
            f"<ul>{items}</ul></div>")


def views_note(current: str | None = None) -> None:
    st.markdown(views_note_html(current), unsafe_allow_html=True)


def answer_body(ans) -> None:
    a = ans.parsed.get("answer")
    if isinstance(a, list):
        st.markdown("\n".join(f"{i}. {md_with_chips(x)}" for i, x in enumerate(a, 1)))
    elif isinstance(a, bool):
        st.markdown(f"**{'Yes' if a else 'No'}**")
    elif a is not None:
        st.markdown(md_with_chips(str(a)))
    if ans.parsed.get("reasoning"):
        st.caption("Reasoning: " + str(ans.parsed["reasoning"]))
