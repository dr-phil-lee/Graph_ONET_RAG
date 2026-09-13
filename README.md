# O*NET Graph RAG Chatbot

A chatbot that answers occupational questions by **traversing the O*NET
knowledge graph** instead of similarity search, and cites every claim to
individual database records (edge-level citations). Built on the S2
graph-RAG system from the ONET-MHQA benchmark study.

**Self-contained**: this repository bundles the retrieval package
(`src/onetrag/`), the prebuilt knowledge graph (`data/graph/`, 30,328 nodes /
236,125 edges over 894 occupations from O*NET 30.3), and the O*NET
level-anchor file (`data/onet/`). The generator is any OpenRouter-hosted
model — you only need an API key.

## Quickstart

```bash
git clone https://github.com/dr-phil-lee/ONET-Graph-RAG.git
cd ONET-Graph-RAG
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # put your OpenRouter key in .env
streamlit run app.py                  # opens http://localhost:8501
```

Get an API key at https://openrouter.ai and set
`OPENROUTER_API_KEY=sk-or-...` in `.env` (gitignored), or paste it in the
app sidebar. Keys are never written anywhere else.

Headless smoke test: `python test_chat.py` (asks three questions and checks
grounded, cited answers come back).

## Features

- **Chat** — ask in English about occupations, tasks, skills, activities.
  Each answer shows citations, metrics, an evidence table (cited lines
  first), and an interactive **evidence subgraph** (red = cited edges);
  click any node or edge to inspect its full attributes.
- **Entity matching** — occupations named exactly are linked directly;
  free-form words ("teachers", "data analyst") are fuzzy-matched against
  894 titles + 48,752 alternate titles. Pin occupations in the sidebar to
  override; pin two to explore how a pair of jobs is connected.
- **Graph Explorer** — node/edge statistics plus an ego-network view around
  any occupation, skill, ability, knowledge area, or activity.
- **PDF report** (header button) — one click saves every answer in the
  session with evidence figures and tables.
- **Model choice** (sidebar) — any model in the OpenRouter catalog
  (default `anthropic/claude-haiku-4.5`).

## How it works

Retrieval is pure graph traversal (no vector database): entity linking →
priority rules (seed-connecting edges, task roll-up chains, descriptor-domain
ratings) → typed expansion (a bridge mode for two-occupation questions,
breadth-first otherwise, capped at 120 edges) → serialization into numbered
evidence lines under a token budget. The generator must answer as JSON and
may cite only the evidence ids it was shown, so every citation resolves to a
real database record.

## Data attribution

O*NET 30.3 Database (May 2026) by the U.S. Department of Labor, Employment
and Training Administration (USDOL/ETA), used under the CC BY 4.0 license.
O*NET® is a trademark of USDOL/ETA. This project is not reviewed or endorsed
by USDOL/ETA.
