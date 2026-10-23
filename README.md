# Mythology & Superhero Bot

A FastAPI service with one endpoint, `POST /ask`, that answers questions about **Norse and Greek mythology**
(49 Wikipedia articles, stored locally) and **comic-book characters** (the live [Superhero API](https://superheroapi.com)).
The LLM picks the source, or both ("How does Marvel's Hercules compare to the Greek Heracles?"), and every
answer lists where its information came from.

## Setup

Python 3.11+, an API key for any OpenAI-compatible LLM (tested with Groq and Gemini), and a Superhero API token
(sign in with GitHub at superheroapi.com).

```bash
cp .env.example .env        # fill in the keys
uv sync --extra dev         # or: python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

## Run

```bash
uv run uvicorn app.main:app --reload
# or: docker build -t myth-bot . && docker run --env-file .env -p 8000:8000 myth-bot
```

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is Batman'\''s intelligence and who publishes him?"}'
```

```json
{
  "answer": "Batman’s intelligence score is 100, and he is published by DC Comics[S2].",
  "sources": [{"id": "S2", "type": "superhero_api", "title": "Batman (Superhero API #70)", "url": null}],
  "tools_used": ["get_superhero"],
  "warnings": []
}
```

Interactive docs at http://localhost:8000/docs.

## Tests and benchmark

```bash
uv run pytest                                  # unit tests, no keys or network needed
uv run python -m benchmark.validate            # checks every benchmark answer exists in the data
uv run python -m benchmark.run                 # 101 live questions, report in benchmark/results/
```

The benchmark covers mythology, superhero, "both", out-of-scope and edge cases (misspellings, missing heroes,
unanswerable questions, a prompt injection). It went from 77/101 to 93/101 as it exposed bugs, and the final
fixes passed 23 of 24 targeted re-checks.

## Design in brief

- **Routing**: the LLM gets two tools, `search_mythology` and `get_superhero`, in a hand-written loop (`app/agent.py`).
- **Retrieval**: BM25 over section-level chunks, behind a `Retriever` interface so embeddings can be added later.
- **Sources**: tool results are labelled `[S1]`, `[S2]`…; only ids the server issued reach the response, so the model
  can't invent a source. Answers given without tools are attributed to `llm`.
- **Resilience**: an optional fallback LLM. On a rate limit or outage the whole question restarts there, because one
  provider's tool-call history is not valid input for another.

## Shortcuts taken

- No conversation memory, auth or rate limiting; cache and index are in memory.
- The Superhero API only searches by name, so "who is the strongest" questions can't be answered.
- Free tiers limit throughput: about 75 questions per day per Groq model (a question is ~2,500 tokens), and ~20
  requests per day on Gemini's free tier. A second Groq model makes the most useful free fallback.

## Data

`data/mythology/` holds Wikipedia articles (CC BY-SA 4.0), fetched with `scripts/fetch_mythology.py`.
