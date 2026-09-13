"""O*NET Graph RAG chatbot — Streamlit UI.

Run:  streamlit run app.py   (from the Chatbot folder, inside .venv)
"""
from __future__ import annotations

import os

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
import streamlit as st
import streamlit.components.v1 as components

import graph_view
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

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {gap: .25rem;}
.stTabs [data-baseweb="tab"] {font-size: .98rem; font-weight: 600;
  padding: .6rem 1.15rem;}
.stTabs [aria-selected="true"] {color: #2563EB;}
.stTabs [data-baseweb="tab-highlight"] {background-color: #2563EB;}

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
    import requests

    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        r.raise_for_status()
        ids = sorted(m["id"] for m in r.json().get("data", []))
        head = [m for m in _POPULAR_MODELS if m in ids]
        return head + [m for m in ids if m not in head]
    except Exception:
        return list(_POPULAR_MODELS)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("O*NET Graph RAG")
    st.caption("Answers cite O*NET 30.3 knowledge-graph edges. English questions "
               "work best.")

    st.subheader("Model (OpenRouter)")
    or_key = st.text_input(
        "API key", type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Get one at openrouter.ai. Stored only in this session "
             "(or in Chatbot/.env as OPENROUTER_API_KEY).")
    model_options = openrouter_models() + [_CUSTOM]
    default_ix = (model_options.index(bootstrap.OPENROUTER_DEFAULT_MODEL)
                  if bootstrap.OPENROUTER_DEFAULT_MODEL in model_options else 0)
    or_model = st.selectbox("Model", model_options, index=default_ix,
                            help="Type to search the OpenRouter catalog. "
                                 "Popular models are listed first.")
    if or_model == _CUSTOM:
        or_model = st.text_input("Custom model id",
                                 value=bootstrap.OPENROUTER_DEFAULT_MODEL)
    if st.button("Apply model", width="stretch"):
        if not or_key:
            st.error("Enter an API key first.")
        else:
            st.session_state.backend_desc = engine.set_backend(or_key, or_model)
    if not engine.is_configured:
        st.warning("Enter your OpenRouter API key and click **Apply model** "
                   "to start chatting.")
    else:
        st.caption("Active: "
                   f"**{st.session_state.get('backend_desc', engine.backend_description)}**")

    st.subheader("Retrieval")
    forced_labels = st.multiselect(
        "Pin occupations (optional)", [t for t, _ in engine.occupations],
        help="Free-form questions can miss the entity linker; pinned "
             "occupations always seed the graph search. Pin two to explore "
             "how a pair of jobs is connected.")
    _title_to_id = dict(engine.occupations)
    forced_seeds = [_title_to_id[t] for t in forced_labels]
    budget = st.slider("Evidence budget (tokens)", 1000, 4000,
                       bootstrap.DEFAULT_CONTEXT_BUDGET, step=500,
                       help="How much graph evidence is placed in front of "
                            "the model per question.")
    engine.set_budget(budget)

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
    knowledge graph and cited to individual database records — no
    unsourced claims.</div>
  <div class="chips">
    <span class="chip">O*NET 30.3 · May 2026</span>
    <span class="chip">{_n_nodes:,} nodes</span>
    <span class="chip">{_n_edges:,} edges</span>
    <span class="chip">894 occupations</span>
    <span class="chip">Edge-level citations</span>
  </div>
</div>
""",
        unsafe_allow_html=True)
with _pdf:
    _answers = [m["answer"] for m in st.session_state.get("messages", [])
                if m.get("answer") is not None]
    if _answers:
        _sig = (len(_answers), tuple(a.question for a in _answers))
        if st.session_state.get("pdf_sig") != _sig:
            import report

            with st.spinner("Preparing PDF..."):
                st.session_state.pdf_bytes = report.build_pdf(
                    _answers, engine,
                    st.session_state.get("backend_desc",
                                         engine.backend_description),
                    engine.rag.budget)
                st.session_state.pdf_sig = _sig
        import datetime as _dt

        st.download_button(
            f"📄 PDF Report ({len(_answers)})",
            data=st.session_state.pdf_bytes,
            file_name=f"onet_graphrag_report_{_dt.date.today():%Y%m%d}.pdf",
            mime="application/pdf", width="stretch",
            help="Save all answers in this session as a formatted PDF report.")
    else:
        st.button("📄 PDF Report", disabled=True, width="stretch",
                  help="Ask a question first — then one click saves the report.")
    if st.button("🗑️ Clear chat", width="stretch",
                 disabled=not st.session_state.get("messages"),
                 help="Delete every question and answer in this session "
                      "and start fresh."):
        st.session_state.messages = []
        st.session_state.pop("pdf_bytes", None)
        st.session_state.pop("pdf_sig", None)
        st.rerun()

# ------------------------------------------------------------------- tabs
tab_chat, tab_explore = st.tabs(["💬 Chat", "🕸️ Graph Explorer"])


def render_answer(ans: Answer):
    if ans.parsed is None:
        st.warning("The model reply was not valid JSON; raw output below.")
        st.code(ans.raw)
        return
    if ans.parsed.get("_truncated"):
        st.info("⚠️ The reply hit the output-token limit and was cut off — "
                "showing the recovered partial answer. The list below may be "
                "incomplete.")
    a = ans.parsed.get("answer")
    if isinstance(a, list):
        st.success("**Answer**\n" + "\n".join(f"- {x}" for x in a))
    else:
        st.success(f"**Answer:** {a}")
    if ans.parsed.get("reasoning"):
        st.markdown(f"*{ans.parsed['reasoning']}*")

    n_cited = sum(1 for _, _, c in ans.evidence if c)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence lines", len(ans.evidence))
    c2.metric("Cited by model", n_cited)
    c3.metric("Context tokens", ans.context_tokens)
    c4.metric("Latency", f"{ans.usage.get('latency_s', 0):.1f}s")

    with st.expander(f"Evidence table ({len(ans.evidence)} lines, "
                     "cited lines first)"):
        import pandas as pd

        rows = sorted(ans.evidence, key=lambda t: not t[2])
        df = pd.DataFrame(
            [{"Cited": "✅" if cited else "",
              "ID": uid,
              "Statement": (text[len(uid) + 3:]
                            if text.startswith(f"[{uid}] ") else text)}
             for uid, text, cited in rows])
        st.dataframe(df, hide_index=True, width="stretch", height=380)
    if ans.retrieved_edge_ids:
        with st.expander("Evidence subgraph — how the answer connects in the graph",
                         expanded=True):
            edges = [engine.edge_index[e] for e in ans.retrieved_edge_ids
                     if e in engine.edge_index]
            html = graph_view.render_edges_html(
                engine.G, edges, engine.anchors,
                cited_edge_ids=set(ans.cited_edge_ids))
            components.html(html, height=570, scrolling=False)
            st.caption("Red = cited by the model · grey = retrieved only. "
                       "Node colors: 🔴 Job · 🔵 Task · 🟢 DWA/IWA/GWA · "
                       "🟠 skills · 🟣 software. **Click any node or edge to "
                       "inspect its full attributes**; drag to rearrange.")
    u = ans.usage
    st.caption(f"context {ans.context_tokens} tok · in {u.get('input_tokens')} / "
               f"out {u.get('output_tokens')} tok · {u.get('latency_s', 0):.1f}s")


with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("answer") is not None:
                if msg.get("entities"):
                    st.caption(msg["entities"])
                render_answer(msg["answer"])
            else:
                st.markdown(msg["content"])

    if question := st.chat_input("Ask about occupations, tasks, skills, activities..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            seeds_preview = engine.preview_seeds(question)
            names = [f"{name} ({label})" for _, name, label in seeds_preview]
            for sid in forced_seeds:
                n, l = engine.node_info(sid)
                names.insert(0, f"📌 {n} ({l})")

            # Free-form words like "teachers"/"drivers" defeat the exact-match
            # linker; fuzzy title matching fills whatever occupations the
            # linker and the sidebar pins did not already cover.
            linked_jobs = [nid for nid, _, label in seeds_preview
                           if label == "Job"]
            auto = engine.auto_occupations(
                question, exclude=forced_seeds + linked_jobs)
            ask_seeds = list(forced_seeds) + [nid for nid, _ in auto]
            names += [f"≈ {title}" for _, title in auto]
            entity_caption = ""
            if names:
                entity_caption = ("Graph entities: " + " · ".join(names) +
                                  ("  — ≈ means auto-matched from your wording; "
                                   "pin different occupations in the sidebar "
                                   "if wrong." if auto else ""))
                st.caption(entity_caption)
                if auto:
                    sugg = engine.suggest_occupations(question)
                    if any(len(v) > 1 for v in sugg.values()):
                        with st.expander("Other occupations matching your words"):
                            for term, cands in sugg.items():
                                st.markdown(f"**{term}** → " +
                                            " · ".join(t for _, t in cands))
            else:
                st.warning("No graph entities matched — the answer will be "
                           "ungrounded. Pin occupations in the sidebar, or "
                           "name them explicitly (e.g. 'Registered Nurses').")
            with st.spinner("Retrieving evidence + generating..."):
                try:
                    ans = engine.ask(question, forced_seeds=ask_seeds)
                except Exception as e:  # missing key / network errors surface here
                    st.error(f"Generation failed: {e}")
                    ans = None
            if ans is not None:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "", "answer": ans,
                     "entities": entity_caption})
                # re-render from history so the header PDF button includes
                # the answer that was just produced
                st.rerun()

with tab_explore:
    meta = graph_view.graph_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Nodes", f"{meta.get('nodes_total', sum(meta.get('nodes_by_label', {}).values())):,}")
        st.dataframe({"node type": list(meta.get("nodes_by_label", {})),
                      "count": list(meta.get("nodes_by_label", {}).values())},
                     height=240, width="stretch")
    with c2:
        st.metric("Edges", f"{meta.get('edges_total', sum(meta.get('edges_by_type', {}).values())):,}")
        st.dataframe({"edge type": list(meta.get("edges_by_type", {})),
                      "count": list(meta.get("edges_by_type", {}).values())},
                     height=240, width="stretch")

    st.divider()
    st.subheader("Ego-network explorer")

    SELECTABLE = ["Job", "EssentialSkill", "TransferableSkill", "Ability",
                  "Knowledge", "WorkStyle", "GWA", "IWA", "DWA"]
    col_a, col_b = st.columns([1, 2])
    with col_a:
        node_label = st.selectbox("Node type", SELECTABLE)
    nodes = sorted((engine.G.nodes[n].get("name", n), n) for n in engine.G
                   if engine.G.nodes[n].get("label") == node_label)
    with col_b:
        picked = st.selectbox(f"{node_label} ({len(nodes):,})",
                              nodes, format_func=lambda t: t[0])

    col1, col2, col3 = st.columns(3)
    with col1:
        radius = st.radio("Radius", [1, 2], horizontal=True)
    with col2:
        etypes = st.multiselect("Edge types", list(meta.get("edges_by_type", {})),
                                default=["HAS_TASK", "MAPS_TO", "PART_OF", "RATED"])
    with col3:
        max_edges = st.slider("Max edges", 50, 300, 150, step=50)

    if picked and etypes:
        center = picked[1]
        edges = graph_view.ego_edges(engine.G, center, radius=radius,
                                     edge_types=set(etypes), max_edges=max_edges)
        total = engine.G.degree(center)
        st.caption(f"Showing {len(edges)} edges around **{picked[0]}** "
                   f"(node has {total} incident edges in total).")
        if edges:
            html = graph_view.render_edges_html(engine.G, edges, engine.anchors)
            components.html(html, height=620, scrolling=False)
            st.caption("**Click any node or edge to inspect its full "
                       "attributes** (description, Job Zone, alternate "
                       "titles, ratings, connectivity); drag to rearrange.")
        else:
            st.info("No edges of the selected types at this radius.")

# ------------------------------------------------------------------ footer
st.divider()
st.markdown(
    '<div class="app-footer">Data: O*NET 30.3 Database (May 2026), '
    "U.S. Department of Labor / ETA, used under CC BY 4.0 — O*NET® is a "
    "trademark of USDOL/ETA. This tool has not been reviewed or endorsed by "
    "USDOL/ETA. · Retrieval: graph-traversal RAG with deterministic, "
    "record-level citations.</div>",
    unsafe_allow_html=True)
