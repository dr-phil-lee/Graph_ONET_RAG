"""O*NET Graph RAG chatbot — Streamlit UI.

Run:  streamlit run app.py   (from the Chatbot folder, inside .venv)

The ChatEngine (graph + linker) is a process-wide cached resource; the
OpenRouter key, model and evidence budget live in st.session_state and are
passed per question, so two browser sessions never share a key.
"""
from __future__ import annotations

import html as _html
import os

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
import streamlit as st

import ai_exposure
import answer_view as av
from engine import Answer, ChatEngine

st.set_page_config(page_title="O*NET Graph RAG", page_icon="🧭",
                   layout="wide")

ACCENT = "#2563EB"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- global polish ---- */
html, body, [class*="css"] {font-family: "Inter", "Segoe UI", sans-serif;}
.block-container {padding-top: 1.2rem; max-width: 1250px;}
h1, h2, h3 {font-family: "Inter", "Segoe UI", sans-serif;
  letter-spacing: -0.01em;}

/* ---- hero header ---- */
.hero {
  background:
    radial-gradient(1100px 240px at 12% -40%, rgba(56,189,248,.28), transparent 60%),
    linear-gradient(120deg, #0B1F33 0%, #10304F 62%, #123A5E 100%);
  color: #ffffff; padding: 1.35rem 1.7rem 1.25rem; border-radius: 16px;
  border: 1px solid rgba(148,197,255,.18);
  box-shadow: 0 8px 26px rgba(9, 30, 55, 0.28);
}
.hero .brand {display:flex; align-items:baseline; gap:.6rem;}
.hero .title {font-size: 1.62rem; font-weight: 800; letter-spacing: -.02em;
  line-height: 1.25;}
.hero .title .thin {font-weight: 400; opacity: .8;}
.hero .tagline {font-size: .95rem; opacity: .82; margin-top: .2rem;
  line-height: 1.5; max-width: 46rem;}
.hero .chips {margin-top: .7rem; display:flex; flex-wrap:wrap; gap:.4rem;}
.hero .chip {font-size: .74rem; font-weight: 600; letter-spacing: .4px;
  text-transform: uppercase; color: #BAD9FF;
  background: rgba(96,165,250,.14); border: 1px solid rgba(96,165,250,.4);
  padding: .16rem .62rem; border-radius: 999px;}

/* ---- answer views: tabs drawn as clickable pills ---- */
.view-hint {display:flex; align-items:center; gap:.55rem; flex-wrap:wrap;
  margin: 1rem 0 .35rem; padding: .55rem .85rem; border-radius: 12px;
  background: linear-gradient(90deg, rgba(37,99,235,.10), rgba(37,99,235,.03));
  border: 1px solid rgba(37,99,235,.28); font-size: .9rem;}
.view-hint b {color: #1D4ED8;}
.view-hint .arrow {font-size: 1.15rem; animation: nudge 1.4s ease-in-out infinite;}
@keyframes nudge {0%,100% {transform: translateY(0);} 50% {transform: translateY(4px);}}
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {gap: .45rem; flex-wrap: wrap;
  border-bottom: 0; padding-bottom: .35rem;}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {font-size: .95rem; font-weight: 650;
  padding: .45rem 1rem; height: auto; border-radius: 999px;
  border: 1.5px solid rgba(37,99,235,.35); background: rgba(37,99,235,.06);
  transition: background .15s, box-shadow .15s, transform .15s;}
.stTabs [data-baseweb="tab"]:hover, .stTabs [role="tab"]:hover {background: rgba(37,99,235,.16);
  box-shadow: 0 2px 8px rgba(37,99,235,.22); transform: translateY(-1px);}
.stTabs [aria-selected="true"] {color: #fff !important; background: #2563EB !important;
  border-color: #2563EB; box-shadow: 0 2px 10px rgba(37,99,235,.35);}
.stTabs [aria-selected="true"] p {color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
  .stTabs .react-aria-SelectionIndicator {display: none;}

/* ---- buttons ---- */
.stDownloadButton button {background: #1D4ED8; color: #fff; border: 0;
  font-weight: 600; box-shadow: 0 2px 8px rgba(29,78,216,.28);}
.stDownloadButton button:hover {background: #2563EB; color: #fff;}
.stButton button[kind="secondary"] {border-radius: 8px;}

/* ---- metrics ---- */
div[data-testid="stMetric"] {background: rgba(37,99,235,.05);
  border: 1px solid rgba(37,99,235,.16); border-radius: 12px;
  padding: .5rem .75rem;}
div[data-testid="stMetric"] label {color: #5B7290; font-size: .78rem;
  text-transform: uppercase; letter-spacing: .5px;}
div[data-testid="stMetricValue"] {font-weight: 700;}

/* ---- chat & content cards ---- */
div[data-testid="stChatMessage"] {border-radius: 14px;}
details {border-radius: 10px;}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] h1 {font-size: 1.18rem; color: #1D4ED8;
  letter-spacing: -.01em;}
section[data-testid="stSidebar"] h3 {font-size: .8rem; text-transform: uppercase;
  letter-spacing: .8px; color: #5B7290; margin-top: .6rem;}

/* ---- "how it is built" panel ---- */
.how {border: 1px solid rgba(120,130,150,.25); border-radius: 16px;
  padding: 1rem 1.1rem .8rem; margin: .3rem 0 1rem; background: rgba(37,99,235,.03);}
.how .h {font-size: 1.05rem; font-weight: 750; margin-bottom: .15rem;}
.how .sub {font-size: .86rem; color: #5B6B82; line-height: 1.5; margin-bottom: .75rem;}
.how .steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: .6rem;}
.how .step {border: 1px solid rgba(120,130,150,.22); border-radius: 12px;
  padding: .65rem .75rem; background: rgba(255,255,255,.55); font-size: .8rem;
  line-height: 1.45; color: inherit;}
.how .step .n {display:inline-flex; width:1.35rem; height:1.35rem; border-radius:50%;
  background:#2563EB; color:#fff; font-size:.72rem; font-weight:700;
  align-items:center; justify-content:center; margin-right:.35rem;}
.how .step .t {font-weight: 700; font-size: .88rem;}
.how .step .d {margin-top: .3rem; color: #475569;}
.how .schema {margin-top: .75rem; font: .76rem/1.6 ui-monospace, Consolas, monospace;
  background: rgba(15,23,42,.05); border-radius: 10px; padding: .5rem .7rem;
  white-space: pre-wrap; color: #334155;}
.how .src {margin-top: .5rem; font-size: .74rem; color: #6B7A90;}
@media (prefers-color-scheme: dark) {
  .how .step {background: rgba(255,255,255,.03);}
  .how .step .d, .how .sub, .how .schema {color: #CBD5E1;}
}

/* ---- footer ---- */
.app-footer {font-size: .78rem; color: #8A97A8; line-height: 1.6;
  padding-top: .2rem;}
</style>
""",
    unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge graph + entity linker...")
def get_engine() -> ChatEngine:
    return ChatEngine()


engine = get_engine()
st.markdown(av.CSS, unsafe_allow_html=True)

# Curated fallback shown when the live model list can't be fetched.
_POPULAR_MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat-v3.1",
    "mistralai/mistral-large-2512",
]
_CUSTOM = "✏️ Enter a model id manually..."


@st.cache_data(ttl=3600, show_spinner=False)
def openrouter_models() -> list[str]:
    """Live model list from OpenRouter (no key needed); popular ones first."""
    from openrouter_llm import list_models

    try:
        ids = list_models()
        head = [m for m in _POPULAR_MODELS if m in ids]
        return head + [m for m in ids if m not in head]
    except Exception:
        return list(_POPULAR_MODELS)


# -------------------------------------------------------- session defaults
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("or_key", os.getenv("OPENROUTER_API_KEY", "").strip())
ss.setdefault("or_model", bootstrap.OPENROUTER_DEFAULT_MODEL)
ss.setdefault("use_memory", True)
ss.setdefault("memory_from", 0)      # answers before this index are forgotten

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("O*NET Graph RAG")
    st.caption("Answers cite O*NET 30.3 knowledge-graph edges. English questions "
               "work best.")

    st.subheader("Model (OpenRouter)")
    key_input = st.text_input(
        "API key", type="password", value=ss.or_key,
        help="Get one at openrouter.ai. Kept only in this browser session "
             "(or in Chatbot/.env as OPENROUTER_API_KEY).")
    model_options = openrouter_models() + [_CUSTOM]
    default_ix = (model_options.index(ss.or_model)
                  if ss.or_model in model_options else len(model_options) - 1)
    model_choice = st.selectbox("Model", model_options, index=default_ix,
                                help="Type to search the OpenRouter catalog. "
                                     "Popular models are listed first.")
    if model_choice == _CUSTOM:
        model_choice = st.text_input("Custom model id", value=ss.or_model).strip()
    if st.button("Apply model", width="stretch"):
        if not key_input.strip():
            st.error("Enter an API key first.")
        elif not model_choice:
            st.error("Enter a model id.")
        else:
            ss.or_key = key_input.strip()
            ss.or_model = model_choice
            st.success(f"Using **{model_choice}**")
    configured = bool(ss.or_key)
    if not configured:
        st.warning("Enter your OpenRouter API key and click **Apply model** "
                   "to start chatting.")
    else:
        st.caption(f"Active: **OpenRouter: {ss.or_model}**")

    st.subheader("Retrieval")
    budget = st.slider("Evidence budget (tokens)", 1000, 4000,
                       bootstrap.DEFAULT_CONTEXT_BUDGET, step=500,
                       help="How much graph evidence is placed in front of "
                            "the model per question.")
    auto_link = st.toggle(
        "Auto-match occupations from wording", value=True,
        help="Fuzzy-match words like 'nurses' or 'data analyst' to O*NET "
             "titles when the exact entity linker finds nothing.")

    st.subheader("🧠 Conversation memory")
    ss.use_memory = st.toggle(
        "Remember the conversation", value=ss.use_memory,
        help="Follow-ups like 'which of those is least exposed to AI?' or "
             "'what does the second one do?' are read with the previous turns: the "
             "question is rewritten into a standalone one and the occupations in "
             "focus carry over. Memory is context only, never cited as evidence.")
    _all_answers = [m["answer"] for m in ss.messages if m.get("answer") is not None]
    _remembered = _all_answers[ss.memory_from:] if ss.use_memory else []
    if _remembered:
        _focus = []
        for _a in _remembered[-4:]:
            for _t in _a.memory_turn()["occupations"]:
                if _t not in _focus:
                    _focus.append(_t)
        st.caption(f"Remembering **{len(_remembered)}** turn(s); the last "
                   f"{min(4, len(_remembered))} are sent with each question.")
        if _focus:
            st.markdown(" ".join(f"<span class='chip'>{_t}</span>" for _t in _focus[:12]),
                        unsafe_allow_html=True)
        if st.button("Forget memory (keep transcript)", width="stretch"):
            ss.memory_from = len(_all_answers)
            st.rerun()
    elif ss.use_memory:
        st.caption("Nothing remembered yet — ask a question.")

backend_desc = f"OpenRouter: {ss.or_model}"

# ---------------------------------------------------------------- header
_head, _pdf = st.columns([4.4, 1], vertical_alignment="center")
with _head:
    _n_nodes = engine.G.number_of_nodes()
    _n_edges = engine.G.number_of_edges()
    st.markdown(
        f"""
<div class="hero">
  <div class="brand">
    <div class="title">🧭 O*NET Graph RAG
      <span class="thin">· Occupational Intelligence</span></div>
  </div>
  <div class="tagline">Ask anything about occupations, tasks, skills, and
    work activities. Every answer is retrieved by traversing the O*NET
    knowledge graph and cited to individual database records — and every
    citation is verified against the evidence the model saw.</div>
  <div class="chips">
    <span class="chip">O*NET 30.3 · May 2026</span>
    <span class="chip">{_n_nodes:,} nodes</span>
    <span class="chip">{_n_edges:,} edges</span>
    <span class="chip">{len(engine.occupations)} occupations</span>
    <span class="chip">Verified citations</span>
    <span class="chip">AI Exposure Index</span>
    <span class="chip">Conversation memory</span>
  </div>
</div>
""",
        unsafe_allow_html=True)
with _pdf:
    _answers = [m["answer"] for m in ss.messages if m.get("answer") is not None]
    if _answers:
        _sig = (len(_answers), tuple(a.question for a in _answers))
        if ss.get("pdf_sig") != _sig:
            import report

            with st.spinner("Preparing PDF..."):
                ss.pdf_bytes = report.build_pdf(_answers, engine, backend_desc, budget)
                ss.pdf_sig = _sig
        import datetime as _dt

        st.download_button(
            f"📄 PDF Report ({len(_answers)})",
            data=ss.pdf_bytes,
            file_name=f"onet_graphrag_report_{_dt.date.today():%Y%m%d}.pdf",
            mime="application/pdf", width="stretch",
            help="Save all answers in this session as a formatted PDF report.")
    else:
        st.button("📄 PDF Report", disabled=True, width="stretch",
                  help="Ask a question first — then one click saves the report.")
    if _answers:
        import json as _json

        st.download_button(
            "💾 Export chat (JSON)",
            data=_json.dumps({"model": backend_desc,
                              "turns": [a.to_dict() for a in _answers]},
                             ensure_ascii=False, indent=2),
            file_name="onet_graphrag_conversation.json", mime="application/json",
            width="stretch",
            help="Every answer with its structured fields, citations, sources, "
                 "AI-exposure rows and evidence.")
    if st.button("🗑️ Clear chat", width="stretch",
                 disabled=not ss.messages,
                 help="Delete every question and answer in this session "
                      "and start fresh (also clears the memory)."):
        ss.messages = []
        ss.memory_from = 0
        for _k in ("pdf_bytes", "pdf_sig", "_html_cache", "pending_question"):
            ss.pop(_k, None)
        st.rerun()

_STATUS_NOTE = {
    "grounded": None,
    "partially_grounded": ("warning", "Some citations the model returned do not "
                                      "exist in the evidence it was shown; only the "
                                      "valid ones are highlighted."),
    "ungrounded": ("warning", "The model cited no evidence, so this answer is "
                              "NOT grounded in O*NET records. Treat it as "
                              "unverified."),
    "truncated": ("info", "⚠️ The reply hit the output-token limit and was cut "
                          "off — showing the recovered partial answer. The list "
                          "below may be incomplete."),
}
_STATUS_BADGE = {"grounded": ("✅ grounded", "#2E7D5B"),
                 "partially_grounded": ("⚠️ partially grounded", "#C9A227"),
                 "ungrounded": ("⛔ ungrounded", "#B23A3A"),
                 "truncated": ("✂️ truncated", "#D9822B"),
                 "unparsed": ("❓ unparsed", "#6B7280")}


def _embed(html: str, height: int) -> None:
    """Interactive pyvis page (graph data only, no model text). st.iframe on
    current Streamlit; components.html on releases that predate it."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # pragma: no cover - older Streamlit
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=False)


def _cached_html(ans: Answer, kind: str, build):
    """pyvis pages are heavy; build each one once per answer."""
    cache = ss.setdefault("_html_cache", {})
    key = (ans.rec.get("qid") or id(ans), kind)
    if key not in cache:
        cache[key] = build()
    return cache[key]


def render_answer(ans: Answer, idx: int, is_last: bool):
    if ans.parsed is None:
        st.warning("The model reply was not valid JSON; raw output below.")
        st.code(ans.raw)
        return
    note = _STATUS_NOTE.get(ans.status)
    if note:
        getattr(st, note[0])(note[1])

    av.methodology(ans)          # how the AI exposure scores were computed
    av.headline(ans)
    av.key_points(ans)
    av.occupation_cards(ans, engine)

    n_cited = sum(1 for _, _, c in ans.evidence if c)
    st.markdown(
        "<div class='view-hint'><span class='arrow'>👇</span><span><b>Six views of "
        "this answer — click each tab to open it:</b> Answer · Focus graph · "
        "Evidence graph · Sources · Evidence · Charts</span></div>", unsafe_allow_html=True)
    t_ans, t_focus, t_full, t_src, t_ev, t_chart = st.tabs([
        "📝 Answer", "🎯 Focus graph", "🕸️ Evidence graph",
        f"📚 Sources ({len(ans.sources)})", f"🔎 Evidence ({len(ans.evidence)})",
        "📊 Charts"])
    with t_ans:
        av.answer_body(ans)
        if ans.invalid_citations:
            st.caption("Citations not found in evidence: " +
                       ", ".join(map(str, ans.invalid_citations)))
    with t_chart:
        av.charts(ans, engine)
    with t_focus:
        html = _cached_html(ans, "focus", lambda: av.focus_graph_html(ans, engine))
        if html:
            _embed(html, 530)
            st.caption("Only what this answer relies on: the records it cited, the "
                       "O*NET Related-Occupations links to each recommended job, and "
                       "every occupation whose profile or AI-exposure score it cited. "
                       "★ = the occupation you asked about; job colors follow the AI "
                       "exposure score (green = lower, red = higher). Click any node "
                       "or edge for details.")
        else:
            st.info("The answer cited no graph records to draw.")
    with t_full:
        html = _cached_html(ans, "full", lambda: av.evidence_graph_html(ans, engine))
        if html:
            _embed(html, 570)
            st.caption("Everything retrieved for this question. Red = cited by the "
                       "model · grey = retrieved only. Node colors: 🔴 Job · 🔵 Task · "
                       "🟢 DWA/IWA/GWA · 🟠 skills · 🟣 software. Click to inspect.")
        else:
            st.info("No graph edges were retrieved.")
    with t_src:
        av.sources(ans)
    with t_ev:
        av.evidence_table(ans)

    label, color = _STATUS_BADGE.get(ans.status, (ans.status, "#6B7280"))
    u = ans.usage
    mem = f" · 🧠 {ans.history_turns} prior turn(s)" if ans.history_turns else ""
    st.markdown(
        f"<span class='pill' style='background:{color}'>{label}</span> "
        f"<span style='color:#6B7A90;font-size:.78rem'>{n_cited}/{len(ans.evidence)} "
        f"evidence lines cited · {ans.model or backend_desc} · context "
        f"{ans.context_tokens} tok · in {u.get('input_tokens')} / out "
        f"{u.get('output_tokens')} tok · {u.get('latency_s', 0):.1f}s{mem}</span>",
        unsafe_allow_html=True)

    if is_last and ans.follow_ups:
        st.markdown("<div class='sec-title'>Ask next</div>", unsafe_allow_html=True)
        cols = st.columns(len(ans.follow_ups))
        for i, fu in enumerate(ans.follow_ups):
            if cols[i].button(fu, key=f"fu-{idx}-{i}", width="stretch"):
                ss.pending_question = fu
                st.rerun()


def memory_history() -> list[dict]:
    """Turns the next question may refer to (after the last 'forget')."""
    if not ss.use_memory:
        return []
    answers = [m["answer"] for m in ss.messages if m.get("answer") is not None]
    return [a.memory_turn() for a in answers[ss.memory_from:]]


@st.cache_data(show_spinner=False)
def how_it_works_html() -> str:
    """Landing-page explainer: how the O*NET graph and the RAG pipeline are
    built. Numbers are read from the loaded graph, not hard-coded."""
    meta = engine.graph_stats()
    nodes = meta.get("nodes_by_label", {})
    edges = meta.get("edges_by_type", {})
    n_alt = sum(len(engine.G.nodes[j].get("alt_titles") or [])
                for _, j in engine.occupations)
    rw = meta.get("rated_weight", {"importance": 0.7, "level": 0.3})
    steps = [
        ("Build the knowledge graph",
         f"The O*NET {meta.get('onet_release', '30.3')} database files (occupations, "
         "tasks and task ratings, detailed/intermediate/generalized work activities, "
         "skills, abilities, knowledge, work styles, related occupations, technology "
         "skills) are turned by a deterministic ETL into one NetworkX graph: "
         f"{meta.get('nodes_total', 0):,} nodes of {len(nodes)} types and "
         f"{meta.get('edges_total', 0):,} edges of {len(edges)} types, covering "
         f"{meta.get('analytic_sample', 0):,} occupations (including those whose skill "
         "ratings O*NET has not published yet, e.g. Data Scientists). Each edge is one "
         "O*NET record with a stable id; rating edges carry importance, level and a "
         f"weight w = {rw['importance']}·IM/5 + {rw['level']}·LV/7. Rows O*NET marks "
         "\"recommend suppress\" are dropped."),
        ("Link the question to the graph",
         f"Words in the question are matched to graph nodes: exact match over "
         f"{len(engine.occupations):,} occupation titles and {n_alt:,} alternate titles "
         "plus skill, task and activity names, then fuzzy matching. If no occupation "
         "is named, the LLM proposes O*NET titles that are matched the same way; "
         "follow-ups reuse the occupations from earlier turns (conversation memory)."),
        ("Traverse and select evidence",
         "From the linked seed nodes the retriever expands a typed subgraph: edges "
         "that join two seeds, task -> activity roll-ups, and ratings ranked by w. "
         "Question intent adds O*NET Related Occupations (career change), occupation "
         "profiles, transferable skills and AI exposure scores. Each record becomes "
         "one numbered evidence line ([E...] record, [N-...] profile, [X-...] AI "
         "exposure) until the token budget is used."),
        ("Generate and verify",
         "An LLM (via OpenRouter) answers only from those evidence lines, as "
         "structured JSON with citations. Every cited id is checked against the "
         "evidence it was shown; occupation cards, Job Zones and AI exposure scores "
         "are rebuilt from the graph and the index, never taken from the model, and "
         "the answer is labelled grounded, partially grounded or ungrounded."),
    ]
    cards = "".join(
        f"<div class='step'><span class='n'>{i}</span><span class='t'>{_html.escape(t)}</span>"
        f"<div class='d'>{_html.escape(d)}</div></div>"
        for i, (t, d) in enumerate(steps, 1))
    schema = ("Job -HAS_TASK-> Task -MAPS_TO-> DWA -PART_OF-> IWA -PART_OF-> GWA\n"
              "Job -RATED{importance, level}-> Skill | Ability | Knowledge | Work style | GWA\n"
              "Skill | Ability | Work style -ENABLES-> GWA      Job -RELATED_TO{rank}-> Job\n"
              "Job -USES_TECH-> Technology      + AI Exposure Index joined on the SOC code")
    top_nodes = ", ".join(f"{k} {v:,}" for k, v in list(nodes.items())[:6])
    top_edges = ", ".join(f"{k} {v:,}" for k, v in list(edges.items())[:5])
    return f"""
<div class="how">
  <div class="h">How this Graph RAG is built</div>
  <div class="sub">Graph RAG = retrieval-augmented generation over a knowledge graph.
    Instead of searching text chunks, the chatbot walks the O*NET occupational graph,
    hands the model only the database records it found, and checks every citation.</div>
  <div class="steps">{cards}</div>
  <div class="schema">{_html.escape(schema)}</div>
  <div class="src">Largest node types: {_html.escape(top_nodes)}. Largest edge types:
    {_html.escape(top_edges)}. Data: O*NET 30.3 Database, U.S. Department of Labor/ETA
    (CC BY 4.0); AI exposure: {_html.escape(ai_exposure.SOURCE_CITATION)}.</div>
</div>"""


# -------------------------------------------------------------------- chat
if not ss.messages:
    st.markdown(how_it_works_html(), unsafe_allow_html=True)
else:
    with st.expander("ℹ️ How this Graph RAG is built"):
        st.markdown(how_it_works_html(), unsafe_allow_html=True)
n_assist = sum(1 for m in ss.messages if m.get("answer") is not None)
seen_assist = 0
for i, msg in enumerate(ss.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("answer") is not None:
            seen_assist += 1
            if msg.get("entities"):
                st.caption(msg["entities"])
            render_answer(msg["answer"], i, is_last=seen_assist == n_assist)
        else:
            st.markdown(msg["content"])

if not ss.messages:
    st.markdown("<div class='sec-title'>Try asking</div>", unsafe_allow_html=True)
    examples = [
        "I was a telemarketer and lost my job to AI. What other jobs fit my skills?",
        "What does a Registered Nurse do, and how exposed is it to AI?",
        "Compare Software Developers and Data Scientists.",
        "I'm an accountant and worried about AI. Which jobs could I switch to?",
    ]
    cols = st.columns(2)
    for j, ex in enumerate(examples):
        if cols[j % 2].button(ex, key=f"ex-{j}", width="stretch"):
            ss.pending_question = ex
            st.rerun()

typed = st.chat_input("Ask about occupations, tasks, skills, AI exposure... "
                      "(follow-ups remember the conversation)")
question = typed or ss.pop("pending_question", None)
if question:
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history = memory_history()
        res = engine.resolve_seeds(question, auto_match=auto_link)
        entity_caption = res.caption()
        if entity_caption:
            st.caption(entity_caption)
        elif history:
            st.caption("🧠 No occupation named — reading the question with the "
                       "conversation so far.")
        elif auto_link:
            st.caption("No occupation named explicitly — asking the model which "
                       "occupation(s) you mean, then searching O*NET for them.")
        else:
            st.warning("No graph entities matched — the model will be told "
                       "there is no evidence. Name the occupation explicitly "
                       "(e.g. 'Registered Nurses').")
        ans = None
        if not configured:
            st.error("Enter your OpenRouter API key in the sidebar and click "
                     "**Apply model** first.")
        else:
            with st.spinner("Retrieving evidence + generating..."):
                try:
                    ans = engine.ask(question,
                                     api_key=ss.or_key, model=ss.or_model,
                                     budget=budget, auto_link=auto_link,
                                     history=history)
                    entity_caption = ans.entity_caption or entity_caption
                except Exception as e:  # bad key / network errors surface here
                    st.error(f"Generation failed: {e}")
        if ans is not None:
            ss.messages.append(
                {"role": "assistant", "content": "", "answer": ans,
                 "entities": entity_caption})
            # re-render from history so the header buttons include it
            st.rerun()

# ------------------------------------------------------------------ footer
st.divider()
st.markdown(
    '<div class="app-footer">Data: O*NET 30.3 Database (May 2026), '
    "U.S. Department of Labor / ETA, used under CC BY 4.0 — O*NET® is a "
    "trademark of USDOL/ETA. This tool has not been reviewed or endorsed by "
    "USDOL/ETA. · AI exposure: " + ai_exposure.SOURCE_CITATION + " (third-party "
    "index, not part of O*NET; methodology: " + ai_exposure.METHOD_URL + "). · Retrieval: graph-traversal RAG with "
    "deterministic, record-level citations. · REST API: "
    "<code>uvicorn api:app</code>.</div>",
    unsafe_allow_html=True)
