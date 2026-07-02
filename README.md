# SHL Conversational Assessment Recommender

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your real GROQ_API_KEY into .env
python scripts/build_catalog.py # builds data/catalog.json from the raw scrape
```

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET  http://localhost:8000/health` → `{"status": "ok"}`
- `POST http://localhost:8000/chat` → see API spec in the assignment PDF

## Test

```bash
# Unit tests for retrieval/state/agent logic (no API key needed, uses mocks)
python -m pytest tests/ -v          # if you add pytest-based tests

# Replay the 10 provided traces against the real agent (needs GROQ_API_KEY)
python -m tests.run_eval

# Targeted guardrail/edge-case probes (needs GROQ_API_KEY)
python -m tests.behavior_probes
```

## Deploy

**Option A: Render** — push to GitHub, create a new Web Service from `render.yaml`,
set `GROQ_API_KEY` in the dashboard.

**Option B: Any Docker host (Fly, Railway, Modal, HF Spaces)** —
```bash
docker build -t shl-agent .
docker run -p 8000:8000 -e GROQ_API_KEY=xxx shl-agent
```

After deploying, confirm both endpoints are reachable at the public URL before submitting.

## Project structure

```
app/
  retrieval.py   # embeddings/TF-IDF search over the catalog
  state.py       # reconstructs conversation state from the stateless message history
  agent.py       # clarify/recommend/refine/compare/refuse decision logic + guardrails
  main.py        # FastAPI /health and /chat
  env.py         # loads .env
data/
  catalog.json   # cleaned catalog (377 individual test solutions)
scripts/
  build_catalog.py  # raw scrape -> data/catalog.json
tests/
  traces/           # the provided C1-C10.md conversation traces
  parse_traces.py   # extracts user turns + expected shortlist from each trace
  run_eval.py        # replays traces through the agent, computes Recall@10
  behavior_probes.py # guardrail/edge-case checks (off-topic, injection, hallucination, turn cap)
```
