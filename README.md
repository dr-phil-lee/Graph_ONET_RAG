# O*NET Graph RAG Chatbot + API

A chatbot that answers occupational questions by **traversing the O*NET
knowledge graph** instead of similarity search, and cites every claim to
individual database records (edge-level citations). Built on the S2
graph-RAG system from the ONET-MHQA benchmark study.

What an answer looks like (the app and every answer are in English):

- **Confirmed occupations**: when the question names no occupation that
  exists in O*NET under that title ("professor", "receptionist", "I sell
  things over the phone"), the chat shows the closest O*NET occupations and
  asks which to proceed with - or lets you pick any other - before anything
  is generated. Exact titles and alternate titles ("RN", "high school
  teacher") go straight through.
- **AI exposure only when it is asked for**: AI exposure scores enter the
  evidence for questions about AI / automation and for career-change
  questions. For a question about AI the answer ends with what the score
  means (how exposed an occupation's work is to AI-driven task automation),
  how it is calculated (the source index, the nine O*NET 31.0 ratings, the
  formula, each occupation's own calculation) and the references. A plain
  "what does X do" or skill question is answered without scores.
- **Honest about gaps**: when the retrieved records do not cover what was
  asked, the answer says so first and lists no occupations without a cited
  record; an ungrounded reply is flagged as the model's own text.
- **Structured**: a one-sentence headline, up to five cited key points,
  occupation cards (Job Zone, O*NET relatedness rank, top transferable
  skills, AI exposure, the model's one-line "why it fits", sources), the full
  answer, and suggested follow-up questions.
- **Graphical**: AI-exposure bar chart (scores only), a "what drives the
  score" chart, a pivot map (preparation vs. AI exposure), O*NET ratings
  chart, an **answer-focused graph** (only what the answer relied on) next to
  the full evidence graph, and a **conversation map** that joins the focus
  graphs of every turn.
- **AI exposure**: every occupation in focus carries its score in the
  [Opportunity Data Occupation AI Exposure Index](https://opportunitydata.org/ai-exposure-occupations.html)
  (third-party, not O*NET), reported as a 0-1 score only (no category
  labels), and the model cites it like any other record.
- **Conversation memory**: follow-ups such as "which of those is least
  exposed to AI?" or "what does the second one do?" are read with the previous
  turns, in the UI and through the API (server sessions or stateless history).

Two front ends share one engine:

| | Command | URL |
|---|---|---|
| Streamlit chat UI | `streamlit run app.py` | http://localhost:8501 |
| REST API (FastAPI) | `uvicorn api:app --port 8000` | http://localhost:8000/docs |

**Self-contained**: this repository bundles the retrieval package
(`src/onetrag/`, a frozen copy of the research code), the prebuilt knowledge
graph (`data/graph/`, 30,328 nodes / 236,125 edges over 894 occupations from
O*NET 30.3), a chat-only extension (`data/graph_ext/`, +29 occupations, 776
nodes, 4,171 edges; see below), and the O*NET level-anchor file
(`data/onet/`).

**Graph extension.** The research graph keeps only occupations with
published skill, ability, knowledge and work-activity ratings (894). O*NET
30.3 has 29 more real occupations whose ratings are not yet published, e.g.
Data Scientists, Project Management Specialists, Financial and Investment
Analysts, Paramedics, Penetration Testers. `graph_ext.py` adds them with what
O*NET does publish (description, Job Zone, alternate titles, tasks, task-to-DWA
mappings, Related Occupations in both directions, technology skills) by
re-running the unchanged research ETL with a wider sample and keeping only
the difference. New edges are numbered from 236,125 on, so every research
edge id keeps its meaning; the research files are not modified. Their profile
lines tell the model that O*NET has no ratings for them. Rebuild with
`python graph_ext.py <path-to-db_30_3_csv>`. The generator is
any OpenRouter-hosted model — you only need an API key.

## Quickstart

```bash
git clone https://github.com/dr-phil-lee/ONET-Graph-RAG.git
cd ONET-Graph-RAG
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # put your OpenRouter key in .env
streamlit run app.py                  # chat UI  -> http://localhost:8501
uvicorn api:app --port 8000           # REST API -> http://localhost:8000/docs
```

Get an API key at https://openrouter.ai and set
`OPENROUTER_API_KEY=sk-or-...` in `.env` (gitignored), paste it in the app
sidebar, or send it per request to the API. Keys are never written anywhere.

Tests:

```bash
python -m pytest -q        # 119 offline tests, fake LLM, no key needed (~2 min)
python test_chat.py        # live smoke test against OpenRouter (needs a key)
```

## Accuracy guarantees

Every answer goes through the same checks, in the UI and the API:

1. **Retrieval is deterministic graph traversal** — no embeddings, no
   vector index. The evidence shown to the model is a list of numbered
   O*NET records (`[E123] Registered Nurses rates the ability "Oral
   Comprehension": importance 4.12/5, level 4.50/7 ...`).
2. **The model may only answer from that evidence.** The chat prompt forbids
   outside knowledge, requires numbers and names to be copied verbatim, and
   asks for an explicit `Insufficient evidence: ...` answer when the graph
   does not contain the fact.
3. **Every citation is verified server-side.** Ids the model returns are
   normalised (`[E12]`, `e12`, `12` → `E12`) and checked against the
   evidence it was actually shown. Invalid ids are reported in
   `invalid_citations`, never silently dropped.
4. **Every answer carries a status**:

   | status | meaning |
   |---|---|
   | `grounded` | valid JSON, ≥ 1 citation, every citation exists in the evidence |
   | `partially_grounded` | some citations were not in the evidence |
   | `ungrounded` | the model cited nothing (e.g. "Insufficient evidence") |
   | `truncated` | the reply hit the output-token limit; partial JSON recovered |
   | `unparsed` | the reply was not JSON; raw text is returned |

5. **Entity matching is conservative, then assisted.** Occupations named
   exactly (or by O*NET-SOC code) are linked directly. Free-form words
   ("nurses", "tele marketer", "registred nurse") are matched against the
   923 titles and 49,579 alternate titles — compound words are joined, typos
   are fuzzy-matched, and a single word must be the *head noun* of the
   title, so "zone" no longer pulls in "Zone Manager" and "marketer" no
   longer pulls in Market Research Analysts. Three-word titles are matched
   as one phrase first ("elementary school teacher" is one occupation, not
   also the Special Education one; "high school teacher" is the O*NET
   alternate title of Secondary School Teachers). When the wording names no
   occupation at all ("I sell things over the phone"), the generator is asked once which occupations the message is
   about and the names are searched in O*NET (`source: "inferred"`). Every
   non-exact entity is marked `≈` in the UI so you can rephrase with the exact title.
6. **Questions are answered as fully as the graph allows.** Career-change,
   "similar job" and "what else could I do" questions pull O*NET's official
   *Related Occupations* (ranked), the profiles of those occupations, and the
   transferable skills / abilities that connect them; "what does X do" pulls
   core tasks plus top-rated skills, abilities and knowledge; technology
   questions pull Technology Skills. The model is told that this evidence is
   sufficient for a recommendation and must answer the parts it can, noting
   only what is genuinely missing. `Insufficient evidence` is reserved for
   questions with no relevant record at all.
7. **Every answer lists its O*NET sources.** Each valid citation is returned
   as a `sources` record with the statement, the O*NET 30.3 table it came
   from (e.g. `Related Occupations (related_occupations.csv)`,
   `Transferable Skills ratings (transferable_skills.csv)`) and the nodes
   involved. The UI shows them under the answer; the PDF report prints them.
8. **Structured fields are checked, not trusted.** An occupation card is kept
   only if that occupation is in the evidence the model saw; its title, Job
   Zone, relatedness rank, skills and AI exposure are read from the graph and
   the index, never from the model. Only the short "fit" sentence is model
   text. Evidence ids written inline in the headline, key points or cards
   count as citations and are validated like the citations array.
9. **AI exposure is never guessed.** Scores come from the Opportunity Data
   index (`data/ai_exposure/`), joined on the 6-digit SOC code **and** the
   title: each index row is scored from one O*NET occupation, so it is
   attached only to the occupation with the same title (e.g. row 15-2099
   "Bioinformatics Technicians" = O*NET 15-2099.01, never 15-2099.00
   "Mathematical Science Occupations, All Other"). All 696 index rows join;
   696 of the 923 graph occupations are covered; for the rest
   (e.g. Chief Executives) the answer says the index does not cover them.
   The index is scored from O*NET 31.0, the graph is O*NET 30.3. Answers
   give the score only; the source site's named bands are not used.
10. **Memory is context, not evidence.** Prior turns are shown to the model
   under "Conversation so far (context only - NOT evidence, never cite it)".
   A follow-up that names no occupation or points back ("those", "the second
   one") is first rewritten into a standalone question, and the
   occupations it refers to become retrieval seeds (`source: "memory"`, 🧠 in
   the UI). Citations must still come from the new evidence.

## AI Exposure Index

The scores come from the Opportunity Data Occupation AI Exposure Index
(methodology v2: O*NET 31.0, BLS OEWS May 2025). The occupation table is
stored at `data/ai_exposure/opportunity_data_ai_exposure_index.csv` (696 SOC
occupations). How a score is built, as documented on the methodology page:

1. Take nine O*NET 31.0 ratings (1-5) per occupation: 5 Work Activities and 4
   Work Context items.
2. Min-max normalize each to 0-1 across the 911 O*NET occupations that have
   all nine; invert "Spend Time Sitting".
3. Average three variables per dimension.
4. Composite = (Digital - Human contact - Physical + 2) / 3, so 0 = fully
   protected and 1 = fully exposed (equal weights). Every row of the stored
   table reproduces this to 4 decimals.

| dimension | O*NET 31.0 variables (element id) | effect |
|---|---|---|
| Digital work intensity | Working with Computers (4.A.3.b.1), Analyzing Data or Information (4.A.2.a.4), Processing Information (4.A.2.a.2) | raises exposure |
| Human contact intensity | Assisting and Caring for Others (4.A.4.a.5), Performing for or Working Directly with the Public (4.A.4.a.8), Face-to-Face Discussions (4.C.1.a.2.l) | lowers exposure |
| Physical task intensity | Spend Time Using Your Hands (4.C.2.d.1.g), Spend Time Sitting, inverted (4.C.2.d.1.a), Responsible for Others' Health and Safety (4.C.1.c.1) | lowers exposure |

Example: Telemarketers `(0.70 - 0.50 - 0.31 + 2) / 3 = 0.63`; Customer
Service Representatives `(0.77 - 0.50 - 0.44 + 2) / 3 = 0.61`.

The chatbot reports the score only; it does not use the source site's named
bands. The index measures task overlap with current AI capability, not job
loss, and the 1:1:1 weighting is a modeling choice.

References (shown under every answer that has a score, and returned as
`ai_exposure_method.references` by the API):

1. Rowe, B. (2026). *Opportunity Data AI Exposure Index.* Opportunity Data.
   Methodology page (methodology v2; released April 2026, updated October
   2026). https://opportunitydata.org/ai-exposure-methodology.html
2. Rowe, B. (2026). *Opportunity Data Occupation AI Exposure Index.*
   Opportunity Data. Occupation table (696 SOC occupations), provided as CSV
   "for research and institutional use".
   https://opportunitydata.org/ai-exposure-occupations.html
3. National Center for O*NET Development. *O*NET 31.0 Database* (August
   2026), Work Activities and Work Context. U.S. Department of Labor, ETA.
   https://www.onetcenter.org/database.html
4. U.S. Bureau of Labor Statistics. *Occupational Employment and Wage
   Statistics (OEWS)*, May 2025 national estimates. https://www.bls.gov/oes/
5. Opportunity Data. *AI Exposure Index vs. BLS AI Exposure Categories*
   (rank correlation 0.76 across 761 occupations).
   https://opportunitydata.org/ai-exposure-bls-comparison.html

In answers the index appears as evidence lines `[X-<soc>]` placed right after
each occupation profile; `sources` records for them name the index, not
O*NET.

## REST API

Interactive docs (OpenAPI/Swagger) are served at `/docs`; the schema at
`/openapi.json`. Each request is independent: the OpenRouter key, model,
evidence budget and pinned occupations you send apply to that call only.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | graph size, default model, whether a server key is set |
| GET | `/models` | OpenRouter model ids (cached 1 h) |
| GET | `/occupations?q=nurse&limit=10` | find O*NET-SOC codes by title / alternate title |
| GET | `/occupations/{soc}` | occupation profile (description, Job Zone, alt titles, edge counts) |
| GET | `/entities?question=...&occupations=...` | dry run: which graph nodes would seed retrieval |
| POST | `/ask` | answer a question with validated citations (+ memory via `session_id`) |
| GET | `/sessions/{id}` | turns the server remembers for one conversation |
| DELETE | `/sessions/{id}` | forget a conversation |
| GET | `/ai-exposure` | index source, methodology, references, score distribution |
| GET | `/ai-exposure/{soc}` | AI exposure of one occupation (404 if not covered) |
| GET | `/graph/stats` | node / edge counts by type |
| GET | `/graph/nodes/{id}` | any node (`29-1141.00`, `4.A.1.a.1`, task ids, ...) |
| GET | `/graph/edges/{id}` | one citable record — resolves the `E123` ids in answers |
| GET | `/graph/ego?node=...&radius=1&edge_types=HAS_TASK` | ego network as JSON |

### Authentication

* **Generator key** (OpenRouter): header `X-OpenRouter-Key: sk-or-...`,
  or body field `api_key`, or the server's `OPENROUTER_API_KEY`. Missing on
  all three → `401`.
* **Server access**: set `CHATBOT_API_TOKEN` in `.env` and every request
  except `/health` must send `Authorization: Bearer <token>`.
* **CORS**: `CHATBOT_CORS_ORIGINS` (comma-separated, default `*`).

### `POST /ask`

```bash
curl -s http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: sk-or-..." \
  -d '{"question": "What is the Job Zone of Registered Nurses?"}'
```

Request body:

```json
{
  "question": "What is the Job Zone of Registered Nurses?",
  "occupations": ["29-1141.00"],      // optional: pin O*NET-SOC codes as seeds
  "auto_link": true,                  // fuzzy-match occupations from wording
  "model": "anthropic/claude-haiku-4.5",
  "budget": 3000,                     // evidence tokens, 500-32000
  "include_evidence": true,           // every evidence line shown to the model
  "include_graph": false,             // evidence subgraph {nodes, edges}
  "include_focus_graph": true,        // answer-focused subgraph, jobs carry ai_exposure
  "session_id": "new",                // server-side memory; reuse the returned id
  "history": null,                    // OR stateless memory: earlier memory_turn objects
  "use_memory": true
}
```

Structured fields in the response (besides the ones shown below):

```json
{
  "headline": "Customer Service Representatives is the closest related job [E199296].",
  "key_points": ["Telemarketers: AI exposure 0.63 [X-41-9041.00]", "..."],
  "occupations": [{"soc": "43-4051.00", "title": "Customer Service Representatives",
                   "job_zone": 2, "related_rank": 2, "related_tier": "Primary-Short",
                   "description": "Interact with customers to provide basic or scripted information ...",
                   "top_skills": [{"name": "Service Orientation", "importance": 4.0, "level": 3.62}],
                   "ai_exposure": {"composite": 0.6107, "digital": 0.7705,
                                   "human_contact": 0.4978, "physical": 0.4408, "match": "exact",
                                   "calculation": "(0.77 - 0.50 - 0.44 + 2) / 3 = 0.61", "...": "..."},
                   "fit": "Same phone-based customer contact [E199296]",
                   "citations": ["E199296", "N-43-4051.00"]}],
  "ai_exposure": [{"soc": "41-9041.00", "title": "Telemarketers", "role": "current",
                   "exposure": {"composite": 0.6291, "calculation": "(0.70 - 0.50 - 0.31 + 2) / 3 = 0.63", "...": "..."}}],
  "follow_ups": ["Which of these is least exposed to AI?", "..."],
  "standalone_question": null,
  "focus_edge_ids": [199296, 199297],
  "focus_graph": {"nodes": ["..."], "edges": ["..."]},
  "session_id": "8f3c...", "session_turns": 1, "history_turns": 0,
  "memory_turn": {"question": "...", "summary": "...", "occupations": ["..."], "seeds": ["..."]}
}
```

Multi-turn with server memory:

```bash
SID=$(curl -s localhost:8000/ask -H "Content-Type: application/json" -H "X-OpenRouter-Key: $KEY" \
  -d '{"question":"I was a telemarketer and lost my job to AI. What else could I do?","session_id":"new"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s localhost:8000/ask -H "Content-Type: application/json" -H "X-OpenRouter-Key: $KEY" \
  -d "{\"question\":\"Which of those is least exposed to AI?\",\"session_id\":\"$SID\"}"
```

Sessions live in memory (LRU 1,000 sessions, 20 turns each, 24 h idle
expiry); set `CHATBOT_SESSION_DIR` to also keep them as JSON files across
restarts.

Response (abridged):

```json
{
  "question": "What is the Job Zone of Registered Nurses?",
  "answer": 4,
  "reasoning": "The occupation profile states Job Zone: 4.",
  "citations": ["N-29-1141.00"],
  "invalid_citations": [],
  "status": "grounded",
  "grounded": true,
  "truncated": false,
  "model": "anthropic/claude-haiku-4.5",
  "entities": [{"node_id": "29-1141.00", "name": "Registered Nurses",
                "label": "Job", "source": "linked", "confidence": "high"}],
  "sources": [{"id": "N-29-1141.00", "kind": "occupation profile",
               "statement": "Occupation: Registered Nurses (O*NET-SOC 29-1141.00). Job Zone: 4. ...",
               "dataset": "O*NET 30.3",
               "table": "Occupation Data + Job Zones (occupation_data.csv, job_zones.csv)",
               "nodes": [{"id": "29-1141.00", "name": "Registered Nurses", "label": "Job"}]}],
  "seeds": ["29-1141.00"],
  "cited_edge_ids": [],
  "retrieved_edge_ids": [8392, 8393, "..."],
  "context_tokens": 2980,
  "usage": {"input_tokens": 3412, "output_tokens": 58, "latency_s": 1.9},
  "evidence": [{"id": "N-29-1141.00", "statement": "Occupation: Registered Nurses (O*NET-SOC 29-1141.00). Job Zone: 4. ...", "cited": true},
               {"id": "E8392", "statement": "Registered Nurses has task (core, importance 4.6/5): \"...\"", "cited": false}],
  "elapsed_s": 2.4
}
```

Evidence ids: `N-<soc>` is an occupation profile line (description, Job
Zone); `E<n>` is one graph edge = one O*NET record, retrievable at
`/graph/edges/<n>`; `X-<soc>` is the occupation's AI Exposure Index line.

Error codes: `401` no OpenRouter key / bad bearer token, `402` OpenRouter
credits exhausted, `404` unknown model or node, `422` validation (unknown
SOC code, empty question, budget out of range), `502` upstream failure after
retries.

### Python client

```python
from client import OnetGraphRagClient

c = OnetGraphRagClient("http://localhost:8000", openrouter_key="sk-or-...")
print(c.search_occupations("nurse", 3))
r = c.ask("Which tasks of Registered Nurses involve administering medication?")
print(r["status"], r["answer"], r["citations"])
for ev in r["evidence"]:
    if ev["cited"]:
        print(ev["id"], ev["statement"])
```

Multi-turn:

```python
chat = c.conversation()                      # server session, created on first ask
r1 = chat.ask("I was a telemarketer and lost my job to AI. What else could I do?")
r2 = chat.ask("Which of those is least exposed to AI?")
print(r2["standalone_question"], [x["title"] for x in r2["ai_exposure"]])
print(c.ai_exposure("41-9041.00"))          # one occupation's index row
chat.forget()
```

`python client.py "your question"` runs a demo against a local server
(reads `OPENROUTER_API_KEY` / `CHATBOT_API_TOKEN` / `CHATBOT_API_URL`).

### JavaScript

```js
const r = await fetch("http://localhost:8000/ask", {
  method: "POST",
  headers: {"Content-Type": "application/json", "X-OpenRouter-Key": "sk-or-..."},
  body: JSON.stringify({question: "What abilities matter most for Pharmacists?"})
});
const {answer, citations, status, evidence} = await r.json();
```

## Chat UI features

- **Landing page** — a plain chat start: the greeting, a two-sentence
  plain-language introduction (an evidence-based chatbot that looks things
  up in the O*NET knowledge graph and answers only from the records it
  finds - Graph RAG) and four example questions.
- **Info bar** (right column, apart from the chat) — two collapsed panels
  that open on click: "How is this Graph RAG built?" (four plain-language
  steps - O*NET becomes a map -> your question is placed on it -> the facts
  around it are collected -> the AI writes from those facts and is checked -
  with counts read from the loaded graph) and "Where does the AI exposure
  score come from?" (what the score means, the source index, the formula,
  the nine O*NET variables and references; when the
  latest answer was about AI, its occupations' own calculations are listed).
- **Occupation confirmation** — if the question names no occupation that
  exists in O*NET under that exact title, the assistant replies with the
  closest O*NET occupations (from the wording, or from the model's reading of
  the message when the wording names none), pre-selects the best one and
  waits: tick the ones to proceed with, pick any other occupation from the
  searchable list, type an occupation name in the chat box, or "Answer
  without choosing". Exact titles, alternate titles, skill/task questions and
  follow-ups ("which of those...") need no confirmation.
- **Chat** — each answer opens directly on six clickable views drawn as
  pill buttons under a "click each tab" hint (📝 Answer · 🎯 Focus graph ·
  🕸️ Evidence graph · 📚 Sources · 🔎 Evidence · 📊 Charts). The Answer view
  holds the whole response: a grounding notice (only when the evidence did
  not cover the question or the model listed occupations without citing a
  record for them), the headline banner, the answer text with citation
  chips, then a card for your occupation and one per recommended occupation
  (Job Zone blocks, an AI exposure bar graph on a 0-1 track, top skills,
  sources, and - for AI / career questions - the difference to your
  current job's score). For questions about AI only, a collapsed "Do you want to
  know about AI exposure score?" panel below the views explains what the
  score means, how it is calculated, each occupation's calculation and the
  references. "Ask next" buttons send a suggested follow-up.
- **Evidence budget** (sidebar) — slider from 1,000 to 10,000 tokens, or
  switch on "Custom amount" and type any value up to 32,000.
- **Conversation memory** (sidebar) — on by default; shows the occupations
  in focus, and "Forget memory" starts fresh without clearing the transcript.
- **Report and Clear chat** (top-right buttons) — the report saves the
  whole session as one PDF: headline, occupations, the full answer,
  sources, the evidence subgraph and table (✓ = cited) and, for AI
  questions, the score methodology last. Clear chat starts a fresh session.
- **Settings panel** — collapsing the sidebar leaves a labelled
  "Open settings panel" button in the top-left corner.
- **Model choice** (sidebar) — any model in the OpenRouter catalog; the key
  and model are kept per browser session.

## How it works

Retrieval is pure graph traversal (no vector database): entity linking →
priority rules (seed-connecting edges, task roll-up chains, descriptor-domain
ratings) → typed expansion (a bridge mode for two-occupation questions,
breadth-first otherwise, capped at 120 edges) → serialization into numbered
evidence lines under a token budget. The generator must answer as JSON and
may cite only the evidence ids it was shown; the engine then verifies each
citation against that evidence.

## Project layout

```
app.py              Streamlit UI
api.py              FastAPI REST API (uvicorn api:app)
client.py           Python client for the API
engine.py           ChatEngine: graph + linker + per-request ask()
hybrid_graph_rag.py chat prompt, intent-aware retrieval, AI-exposure lines, history block
ai_exposure.py      Opportunity Data AI Exposure Index: load, join on SOC, methodology, references
memory.py           conversation turns + ConversationStore (API sessions)
answer_view.py      structured answer rendering: banner, cards, Altair charts, graphs
openrouter_llm.py   OpenRouter backend with retry / error policy
graph_view.py       pyvis evidence subgraph + ego network
report.py           PDF report (build_pdf, the header download) + Markdown report (build_markdown)
bootstrap.py        paths, .env, defaults
src/onetrag/        frozen research package (never edit; tests assert it matches Research/src)
data/graph/         nodes.parquet, edges.parquet, meta.json
data/onet/          level_scale_anchors.csv
data/ai_exposure/   opportunity_data_ai_exposure_index.csv
tests/              offline test suite (pytest)
```

## Data attribution

O*NET 30.3 Database (May 2026) by the U.S. Department of Labor, Employment
and Training Administration (USDOL/ETA), used under the CC BY 4.0 license.
O*NET® is a trademark of USDOL/ETA. This project is not reviewed or endorsed
by USDOL/ETA.

AI exposure scores: Rowe, B. (2026). Opportunity Data Occupation AI Exposure
Index, https://opportunitydata.org/ai-exposure-occupations.html, provided for
research and institutional use. Employment figures in the index are BLS OEWS
national estimates. The index is a third-party estimate and is not part of
O*NET.
