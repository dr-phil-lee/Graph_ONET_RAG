# O*NET Graph RAG Chatbot

Interactive chatbot on top of the S2 graph-RAG system from the ONET-MHQA study.
It imports the frozen research code from `../src/onetrag` unchanged; the
generator is any **OpenRouter-hosted model** (you supply an API key).

Everything else the chatbot needs is local: the knowledge graph
(`../data/graph/*.parquet`, ~4.4 MB) and the O*NET CSVs
(`../../db_30_3_csv`). No vector database is used — S2 retrieval is pure
graph traversal.

> ⚠️ Never edit `../config.yaml` — its hash keys the research LLM cache.
> The chatbot overrides settings in memory only.

## Setup (once)

```powershell
cd Research\Chatbot
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env    # then put your OpenRouter key in .env
```

Get an API key at https://openrouter.ai and set it in `.env`:
`OPENROUTER_API_KEY=sk-or-...` (gitignored; you can also paste the key in
the app sidebar instead). Keys are never written anywhere else.

## Run

```powershell
.\.venv\Scripts\python test_chat.py        # headless smoke test (needs the key)
.\.venv\Scripts\streamlit run app.py       # web UI at http://localhost:8501
```

## Using the app

- **Sidebar → Model (OpenRouter)** — the key is pre-filled from `.env`;
  pick a model from the live catalog (default `anthropic/claude-haiku-4.5`)
  and click **Apply model**. Answers typically take a few seconds.
- **Chat tab** — ask in English. Each answer shows the model's citations,
  metrics, an evidence table (cited lines first), and an interactive
  **evidence subgraph** (red = cited edges). Click any node or edge in a
  graph to inspect its full attributes.
- **Pin occupations** (sidebar) when the entity linker misses your phrasing;
  free-form words like "teachers" are also auto-matched against all titles
  and alternate titles.
- **PDF report** (header button) — one click saves every answer in the
  session: a title page, then per question the answer, reasoning, citations,
  a static evidence-subgraph figure, and the full evidence table.
- **Clear chat** (header button) — wipe the session and start fresh.
- **Graph Explorer tab** — overall node/edge statistics plus an ego-network
  view around any occupation, skill, ability, knowledge area, or activity.
