"""O*NET Graph RAG chatbot — Streamlit UI.

Run:  streamlit run app.py   (from the Chatbot folder, inside .venv)

The ChatEngine (graph + linker) is a process-wide cached resource; the
OpenRouter key, model and evidence budget live in st.session_state and are
passed per question, so two browser sessions never share a key.

Flow of one question:
  1. the occupations in the question are resolved; when none exists in
     O*NET under that exact title, the closest O*NET occupations are shown
     and the user confirms (or picks another) before anything is generated;
  2. evidence is retrieved from the graph and the model answers from it;
  3. the answer is rendered as six views (Answer, Focus graph, Evidence
     graph, Sources, Evidence, Charts); the Answer view holds the grounding
     notice, headline, answer text and occupation cards. Below the views a
     collapsed "Do you want to know about AI exposure score?" panel explains
     the score (only for questions about AI).
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import os

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
import streamlit as st

import ai_exposure
import answer_view as av
from engine import Answer, ChatEngine

ss = st.session_state
st.set_page_config(page_title="GraphRAG-Based O*NET Chatbot", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
<style>
/* ---- layout: one readable column, like a chat app ---- */
.block-container {padding-top: .9rem; padding-bottom: 6rem; max-width: 1340px;}
[data-testid="stBottomBlockContainer"] {max-width: 1340px;}
[data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {max-width: 67%;}
h1, h2, h3 {letter-spacing: -0.01em;}
div[data-testid="stCaptionContainer"] {opacity: 1 !important;}
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p,
  div[data-testid="stCaptionContainer"] *, .stCaption, small {color: #374151 !important;}
div[data-testid="stChatMessage"] {border-radius: 14px; padding: .6rem .9rem;}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
  {background: #F3F4F6;}

/* ---- sidebar: once collapsed, the control that re-opens it is a clearly
   labelled button in the top-left corner (Streamlit's own is a faint icon) ---- */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"]
  {opacity: 1 !important; visibility: visible !important; display: inline-flex !important;
   align-items: center; gap: .3rem; background: #2563EB !important; color: #fff !important;
   border: 0; border-radius: 999px; padding: .3rem .8rem .3rem .55rem;
   box-shadow: 0 2px 10px rgba(37,99,235,.35);}
[data-testid="stExpandSidebarButton"] span, [data-testid="stSidebarCollapsedControl"] span
  {color: #fff !important;}
[data-testid="stExpandSidebarButton"]::after {content: "Open settings panel";
  font-size: .85rem; font-weight: 700; color: #fff; white-space: nowrap;}
section[data-testid="stSidebar"] h3 {font-size: .8rem; text-transform: uppercase;
  letter-spacing: .8px; color: #374151; margin-top: .6rem;}

/* ---- top bar ---- */
.topbar {display:flex; align-items:baseline; gap:.2rem .6rem; padding:.1rem 0 .4rem; flex-wrap:wrap;}
.topbar .t {font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; white-space: nowrap;}
.topbar .s {font-size: .95rem; color: #4B5563; flex-basis: 100%;}
.st-key-topbtn-report button, .st-key-topbtn-clear button {white-space: nowrap;}

/* ---- landing ---- */
.landing {text-align:center; padding: 2.6rem 0 1.4rem;}
.landing .g {font-size: 2.3rem; font-weight: 800; letter-spacing: -.025em; line-height: 1.2;}
.landing .d {font-size: 1.12rem; color: #374151; margin: .8rem auto 0; line-height: 1.6;
  max-width: 46rem;}
.landing .d2 {font-size: 1.12rem; color: #111827; margin-top: .7rem; font-weight: 600;}
.landing .d b {color: #111827;}

/* ---- info bar (right column): reference panels, apart from the chat ---- */
div[data-testid="stColumn"]:has(.st-key-infobar) {position: sticky; top: 3.2rem;
  align-self: flex-start; max-height: calc(100vh - 8.5rem); overflow-y: auto;}
.st-key-infobar {border: 1px solid #D1D5DB; border-radius: 14px; background: #F9FAFB;
  padding: .8rem .85rem .4rem; font-size: .93rem;}
.infobar-title {font-size: 1.05rem; font-weight: 800; color: #111827;}
.infobar-sub {font-size: .86rem; color: #4B5563; line-height: 1.45; margin: .2rem 0 .6rem;}
.st-key-infobar details {background: #fff;}
.st-key-infobar .how .flow {flex-direction: column;}
.st-key-infobar .how .arrow {transform: rotate(90deg); padding: .1rem 0;}
.st-key-infobar .method {margin-top: .2rem;}
.st-key-infobar .method .dims {flex-direction: column;}

/* ---- "how it is built" explainer ---- */
.how .intro {font-size: .95rem; line-height: 1.55; color: #1F2937; margin-bottom: .8rem;}
.how .flow {display:flex; gap:.35rem; align-items:stretch; flex-wrap:wrap;}
.how .st {flex:1 1 150px; border:1px solid #D1D5DB; border-radius:14px; padding:.75rem .8rem;
  background:#F9FAFB; min-width: 150px;}
.how .st .n {display:inline-flex; width:1.6rem; height:1.6rem; border-radius:50%;
  background:#2563EB; color:#fff; font-size:.8rem; font-weight:800; align-items:center;
  justify-content:center; margin-right:.4rem;}
.how .st .ic {font-size:1.35rem; vertical-align:middle;}
.how .st .t {display:block; font-weight:800; font-size:.95rem; margin:.45rem 0 .3rem; color:#111827;}
.how .st .d {font-size:.84rem; line-height:1.5; color:#1F2937;}
.how .arrow {align-self:center; font-size:1.3rem; color:#2563EB; font-weight:800; padding:0;}
.how .src {margin-top:.7rem; font-size:.78rem; color:#4B5563; line-height:1.5;}
@media (max-width: 900px) {.how .arrow {display:none;}}

/* ---- answer views: tabs drawn as clickable pills ---- */
.view-hint {display:flex; align-items:center; gap:.55rem; flex-wrap:wrap;
  margin: 1rem 0 .35rem; padding: .5rem .85rem; border-radius: 12px;
  background: #EFF4FF; border: 1px solid #BFD3FF; font-size: .9rem; color:#111827;}
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

/* ---- confirmation card ---- */
.confirm {border:1px solid #F59E0B; border-left:6px solid #F59E0B; border-radius:14px;
  padding:.75rem 1rem .6rem; background:#FFFBEB; margin:.2rem 0 .6rem; color:#111827;}
.confirm .lbl {font-size:.7rem; font-weight:800; letter-spacing:.9px; text-transform:uppercase;
  color:#B45309;}
.confirm .q {font-size:1rem; font-weight:650; margin:.25rem 0 .2rem;}
.confirm .d {font-size:.9rem; line-height:1.5;}

/* ---- buttons ---- */
.stDownloadButton button {font-weight: 600;}
.stButton button[kind="secondary"] {border-radius: 10px;}

/* ---- metrics ---- */
div[data-testid="stMetric"] {background: rgba(37,99,235,.05);
  border: 1px solid rgba(37,99,235,.16); border-radius: 12px; padding: .5rem .75rem;}

/* ---- footer ---- */
.app-footer {font-size: .78rem; color: #4B5563; line-height: 1.6; padding-top: .2rem;}
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
ss.setdefault("messages", [])
ss.setdefault("or_key", os.getenv("OPENROUTER_API_KEY", "").strip())
ss.setdefault("or_model", bootstrap.OPENROUTER_DEFAULT_MODEL)
ss.setdefault("use_memory", True)
ss.setdefault("memory_from", 0)      # answers before this index are forgotten
ss.setdefault("pending_confirm", None)   # {"question", "terms"} awaiting the user


def _all_answers() -> list[Answer]:
    return [m["answer"] for m in ss.messages if m.get("answer") is not None]


def memory_history() -> list[dict]:
    """Turns the next question may refer to (after the last 'forget')."""
    if not ss.use_memory:
        return []
    return [a.memory_turn() for a in _all_answers()[ss.memory_from:]]


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧭 O*NET Graph RAG")
    st.caption("Answers are built only from O*NET 30.3 records and cited. "
               "English questions work best.")

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
        st.caption(f"Active: **{ss.or_model}**")

    st.subheader("Evidence budget")
    custom_budget = st.toggle(
        "Custom amount", value=False,
        help=f"Type any amount up to {bootstrap.MAX_CONTEXT_BUDGET:,} tokens "
             "instead of using the slider.")
    if custom_budget:
        budget = int(st.number_input(
            "Tokens of O*NET evidence per question",
            min_value=bootstrap.MIN_CONTEXT_BUDGET, max_value=bootstrap.MAX_CONTEXT_BUDGET,
            value=bootstrap.DEFAULT_CONTEXT_BUDGET, step=500))
    else:
        budget = st.slider("Tokens of O*NET evidence per question", 1000,
                           bootstrap.BUDGET_SLIDER_MAX, bootstrap.DEFAULT_CONTEXT_BUDGET,
                           step=500)
    st.caption("More evidence = fuller answers, slower and costlier replies. Your model "
               "must accept the context size.")
    auto_link = st.toggle(
        "Auto-match occupations from wording", value=True,
        help="Match words like 'nurses' or 'data analyst' to O*NET titles. When the "
             "match is not exact you are asked to confirm before the answer is "
             "generated.")

    st.subheader("🧠 Conversation memory")
    ss.use_memory = st.toggle(
        "Remember the conversation", value=ss.use_memory,
        help="Follow-ups like 'which of those is least exposed to AI?' or "
             "'what does the second one do?' are read with the previous turns: the "
             "question is rewritten into a standalone one and the occupations in "
             "focus carry over. Memory is context only, never cited as evidence.")
    _answers_all = _all_answers()
    _remembered = _answers_all[ss.memory_from:] if ss.use_memory else []
    if _remembered:
        _focus = []
        for _a in _remembered[-4:]:
            for _t in _a.memory_turn()["occupations"]:
                if _t not in _focus:
                    _focus.append(_t)
        st.caption(f"Remembering **{len(_remembered)}** turn(s); the last "
                   f"{min(4, len(_remembered))} are sent with each question.")
        if _focus:
            st.markdown(" ".join(f"<span class='chip'>{_html.escape(_t)}</span>"
                                 for _t in _focus[:12]), unsafe_allow_html=True)
        if st.button("Forget memory (keep transcript)", width="stretch"):
            ss.memory_from = len(_answers_all)
            st.rerun()
    elif ss.use_memory:
        st.caption("Nothing remembered yet — ask a question.")

    backend_desc = f"OpenRouter: {ss.or_model}"

# ------------------------------------------- top bar: title · report · clear
_c_title, _c_report, _c_clear = st.columns([5.6, 1.6, 1.3], vertical_alignment="center")
with _c_title:
    st.markdown("<div class='topbar'><span class='t'>🧭 GraphRAG-Based O*NET Chatbot</span>"
                "<span class='s'>Occupations, skills and AI exposure, answered from "
                "O*NET records with verified citations</span></div>",
                unsafe_allow_html=True)
with _c_report, st.container(key="topbtn-report"):
    if _answers_all:
        import report

        st.download_button(
            f"⬇️ Report (PDF, {len(_answers_all)})",
            data=report.build_pdf(_answers_all, engine, backend_desc, budget),
            file_name=f"onet_graphrag_report_{_dt.date.today():%Y%m%d}.pdf",
            mime="application/pdf", width="stretch",
            help="Every answer in this session - headline, occupations, the full "
                 "answer, sources and evidence - as one PDF file.")
    else:
        st.button("⬇️ Report (PDF)", disabled=True, width="stretch",
                  help="Ask a question first — then one click saves the report.")
with _c_clear, st.container(key="topbtn-clear"):
    if st.button("🗑️ Clear chat", width="stretch", disabled=not ss.messages,
                 help="Delete every question and answer in this session "
                      "and start fresh (also clears the memory)."):
        ss.messages = []
        ss.memory_from = 0
        ss.pending_confirm = None
        for _k in ("_html_cache", "pending_question"):
            ss.pop(_k, None)
        st.rerun()


# ------------------------------------------------------------- explainer
@st.cache_data(show_spinner=False)
def how_it_works_html() -> str:
    """Plain-language explainer: four steps, numbers read from the loaded graph."""
    meta = engine.graph_stats()
    n_nodes = meta.get("nodes_total", 0)
    n_edges = meta.get("edges_total", 0)
    n_jobs = len(engine.occupations)
    steps = [
        ("🗺️", "O*NET becomes a map",
         f"O*NET is the U.S. Department of Labor's database of occupations. We load it as "
         f"one connected map: {n_jobs} occupations linked to their tasks, skills, "
         f"abilities, knowledge, related occupations and software — {n_nodes:,} points "
         f"and {n_edges:,} connections. Every connection is one O*NET record with its "
         "own id."),
        ("🔎", "Your question is placed on the map",
         "We find the occupations, skills or activities you mention — by their O*NET "
         "titles, common alternate names (\"RN\", \"high school teacher\") or close "
         "matches. If there is no exact O*NET occupation for what you typed, we show "
         "the closest ones and ask you to confirm before going on."),
        ("🧩", "We collect the facts around them",
         "Starting from those points we walk the map and gather the O*NET records that "
         "matter for your question: tasks, importance ratings, related occupations, and "
         "— when you ask about AI — each occupation's AI exposure score. These records "
         "are the only material the AI model gets."),
        ("✅", "The AI writes from those facts — and we check it",
         "The language model answers using only the collected records and cites each "
         "one. Every citation is verified against the records it was shown, and the "
         "answer is labelled grounded, partially grounded or ungrounded. The sources, "
         "evidence and graphs are open to you under every answer."),
    ]
    cards = []
    for i, (ic, t, d) in enumerate(steps, 1):
        cards.append(f"<div class='st'><span class='n'>{i}</span><span class='ic'>{ic}</span>"
                     f"<span class='t'>{_html.escape(t)}</span>"
                     f"<div class='d'>{_html.escape(d)}</div></div>")
        if i < len(steps):
            cards.append("<div class='arrow'>➜</div>")
    return f"""
<div class="how">
  <div class="intro"><b>Graph RAG</b> means the chatbot does not search the web or rely
    on what the AI model remembers. It looks things up in a map built from the O*NET
    database, hands the model only what it found, and shows you the records.</div>
  <div class="flow">{''.join(cards)}</div>
  <div class="src">Data: O*NET {meta.get('onet_release', '30.3')} Database, U.S. Department
    of Labor / ETA (CC BY 4.0). AI exposure scores: {_html.escape(ai_exposure.SOURCE_CITATION)}
    (a third-party index, not part of O*NET).</div>
</div>"""


_STATUS_BADGE = {"grounded": ("✅ grounded", "#2E7D5B"),
                 "partially_grounded": ("⚠️ partially grounded", "#B7791F"),
                 "ungrounded": ("⛔ ungrounded", "#B23A3A"),
                 "truncated": ("✂️ truncated", "#D9822B"),
                 "unparsed": ("❓ unparsed", "#4B5563")}


def _embed(html: str, height: int) -> None:
    """Interactive pyvis page (graph data only, no model text)."""
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


def render_answer(ans: Answer, idx: int, is_last: bool, entities: str | None = None):
    """The six views come first; everything that makes up the response
    (grounding notice, headline, the answer text, occupation cards) lives
    under the Answer view, so the reply opens on one clean tab strip."""
    if ans.parsed is None:
        st.warning("The model reply was not valid JSON; raw output below.")
        st.code(ans.raw)
        return

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
        if entities:
            st.caption(entities)
        if ans.status == "truncated":
            st.info("✂️ The reply hit the output-token limit and was cut off — showing "
                    "the recovered partial answer. The list below may be incomplete.")
        av.grounding_notice(ans)     # first: is this answer backed by O*NET records?
        av.headline(ans)
        av.answer_body(ans)
        av.occupation_cards(ans, engine)
        if ans.invalid_citations:
            st.caption("Citations not found in evidence: " +
                       ", ".join(map(str, ans.invalid_citations)))
    with t_focus:
        html = _cached_html(ans, "focus", lambda: av.focus_graph_html(ans, engine))
        if html:
            _embed(html, 530)
            colors = ("job colors follow the AI exposure score (green = lower, red = "
                      "higher)" if ans.exposure_in_evidence else "jobs are red")
            st.caption("Only what this answer relies on: the records it cited, the "
                       "O*NET Related-Occupations links to each recommended job, and "
                       "every occupation whose profile it cited. ★ = the occupation you "
                       f"asked about; {colors}. Click any node or edge for details.")
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
    with t_chart:
        av.charts(ans, engine)

    label, color = _STATUS_BADGE.get(ans.status, (ans.status, "#4B5563"))
    u = ans.usage
    mem = f" · 🧠 {ans.history_turns} prior turn(s)" if ans.history_turns else ""
    st.markdown(
        f"<span class='pill' style='background:{color}'>{label}</span> "
        f"<span style='color:#4B5563;font-size:.78rem'>{n_cited}/{len(ans.evidence)} "
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


def run_question(question: str, forced_seeds: list[str] | None, note: str | None = None):
    """Retrieve + generate inside an assistant bubble, then store the answer."""
    with st.chat_message("assistant"):
        if note:
            st.caption(note)
        ans = None
        if not configured:
            st.error("Enter your OpenRouter API key in the sidebar and click "
                     "**Apply model** first.")
        else:
            with st.spinner("Retrieving evidence + generating..."):
                try:
                    ans = engine.ask(question, forced_seeds=forced_seeds,
                                     api_key=ss.or_key, model=ss.or_model,
                                     budget=budget, auto_link=auto_link,
                                     history=memory_history())
                except Exception as e:  # bad key / network errors surface here
                    st.error(f"Generation failed: {e}")
        if ans is not None:
            ss.messages.append({"role": "assistant", "content": "", "answer": ans,
                                "entities": ans.entity_caption})
            st.rerun()
        st.stop()      # generation failed: keep the transcript as it is


def confirmation_card(pending: dict) -> None:
    """Ask which O*NET occupation(s) the question is about before answering."""
    terms = pending["terms"]
    names = " / ".join(f"<b>{_html.escape(t['term'])}</b>" for t in terms)
    with st.chat_message("assistant"):
        st.markdown(
            f"<div class='confirm'><div class='lbl'>Please confirm the occupation</div>"
            f"<div class='q'>O*NET has no occupation called {names} exactly.</div>"
            f"<div class='d'>These are the closest O*NET occupations. Tick the "
            f"ones to proceed with, pick another from the full list, or type a different "
            f"occupation name in the chat box.</div></div>",
            unsafe_allow_html=True)
        picked: list[str] = []
        label_of: dict[str, str] = {}
        for t in terms:
            opts = [soc for soc, _ in t["candidates"]]
            label_of.update({soc: f"{title} ({soc})" for soc, title in t["candidates"]})
            via = ("from your wording" if t["via"] == "wording"
                   else "as the model read your message")
            picked += st.pills(f"Closest to “{t['term']}” ({via})", opts,
                               format_func=lambda s: label_of[s], selection_mode="multi",
                               default=opts[:1], key=f"confirm-{pending['question']}-{t['term']}")
        extra = st.selectbox("Or choose any other O*NET occupation", [None] + engine.occupations,
                             format_func=lambda t: "— type to search —" if t is None else
                             f"{t[0]} ({t[1]})", key=f"confirm-other-{pending['question']}")
        if extra:
            picked.append(extra[1])
        picked = list(dict.fromkeys(picked))
        c1, c2 = st.columns([1, 1])
        go = c1.button(f"✅ Proceed with {len(picked)} occupation(s)" if picked else
                       "✅ Proceed", type="primary", disabled=not picked, width="stretch")
        skip = c2.button("Answer without choosing", width="stretch",
                         help="The model is told the evidence may not match your wording.")
        if go or skip:
            ss.pending_confirm = None
            chosen = picked if go else []
            if chosen:
                titles = ", ".join(engine.G.nodes[s].get("name", s) for s in chosen)
                note = f"📌 Confirmed: {titles}"
            else:
                note = "No occupation confirmed — answering from the closest matches."
            run_question(pending["question"], chosen or None, note)


# -------------------------------------------------------------------- chat
_answers = _all_answers()
chat_col, info_col = st.columns([2.05, 1], gap="large")


# ------------------------------------------- info bar (right): explainers
def latest_ai_answer() -> Answer | None:
    for a in reversed(_answers):
        if getattr(a, "ai_relevant", False) and a.exposure_method:
            return a
    return None


with info_col, st.container(key="infobar"):
    st.markdown("<div class='infobar-title'>ℹ️ About this chatbot</div>"
                "<div class='infobar-sub'>Click a question to open it. These panels are "
                "reference material and stay out of the chat.</div>",
                unsafe_allow_html=True)
    with st.expander("How is this Graph RAG built?", icon="💡"):
        st.markdown(how_it_works_html(), unsafe_allow_html=True)
    with st.expander(av.METHOD_PROMPT, icon="🤖"):
        av.exposure_panel(latest_ai_answer())

# ------------------------------------------------------- chat (left column)
with chat_col:
    if not ss.messages:
        st.markdown(
            "<div class='landing'><div class='g'>What would you like to know about an "
            "occupation?</div>"
            "<div class='d'>This is an <b>evidence-based chatbot</b>. It does not answer "
            "from memory: it looks things up in the <b>O*NET knowledge graph</b> — the "
            "U.S. Department of Labor's map of over 900 occupations and their tasks, "
            "skills and links — and answers only from the records it finds there "
            "(<b>Graph RAG</b>). Every answer cites those records, and you can open "
            "them under the answer.</div>"
            "<div class='d2'>Ask about tasks, skills, related jobs or <b>AI exposure</b>."
            "</div></div>",
            unsafe_allow_html=True)
        examples = [
            "I was a telemarketer and lost my job to AI. What other jobs fit my skills?",
            "What does a Registered Nurse do?",
            "Compare Software Developers and Data Scientists.",
            "I'm an accountant and worried about AI. Which jobs could I switch to?",
        ]
        cols = st.columns(2)
        for j, ex in enumerate(examples):
            if cols[j % 2].button(ex, key=f"ex-{j}", width="stretch"):
                ss.pending_question = ex
                st.rerun()

    n_assist = len(_answers)
    seen_assist = 0
    for i, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("answer") is not None:
                seen_assist += 1
                render_answer(msg["answer"], i, is_last=seen_assist == n_assist,
                              entities=msg.get("entities"))
            else:
                st.markdown(msg["content"])

    if ss.pending_confirm:
        confirmation_card(ss.pending_confirm)

# the chat box stays at top level so Streamlit pins it to the bottom
typed = st.chat_input("Ask about occupations, tasks, skills, AI exposure... "
                      "(follow-ups remember the conversation)")
question = typed or ss.pop("pending_question", None)
if question:
    pending, ss.pending_confirm = ss.pending_confirm, None
    # while a confirmation is open, a short reply that names an occupation
    # answers it ("business professor"); anything else is a new question
    hits = engine.search_occupations(question, 1) if pending else []
    with chat_col:
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        if hits and hits[0]["score"] >= 90 and len(question.split()) <= 6:
            run_question(pending["question"], [hits[0]["soc"]],
                         f"📌 Confirmed: {hits[0]['title']}")
        else:
            proposal = engine.propose_occupations(
                question, history=memory_history(), auto_match=auto_link,
                api_key=ss.or_key or None, model=ss.or_model)
            if proposal["confirm"]:
                ss.pending_confirm = {"question": question, "terms": proposal["terms"]}
                st.rerun()
            run_question(question, None)

# ------------------------------------------------------------------ footer
st.divider()
st.markdown(
    '<div class="app-footer">Data: O*NET 30.3 Database (May 2026), '
    "U.S. Department of Labor / ETA, used under CC BY 4.0 — O*NET® is a "
    "trademark of USDOL/ETA. This tool has not been reviewed or endorsed by "
    "USDOL/ETA. · AI exposure: " + ai_exposure.SOURCE_CITATION + " (third-party "
    "index, not part of O*NET; methodology: " + ai_exposure.METHOD_URL + "). · "
    "Retrieval: graph-traversal RAG with deterministic, record-level citations. · "
    "REST API: <code>uvicorn api:app</code>.</div>",
    unsafe_allow_html=True)
