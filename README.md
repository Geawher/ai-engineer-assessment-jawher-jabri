# Mythology & Superhero Bot

A FastAPI service with one endpoint, `POST /ask`, that answers questions from **document collections**
(today: 49 Wikipedia articles on Norse and Greek mythology) and **comic-book characters** (the live
[Superhero API](https://superheroapi.com)). The LLM picks the source, or both ("How does Marvel's Hercules compare
to the Greek Heracles?"), and every answer lists where its information came from.

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
  "sources": [{"id": "S2", "type": "superhero_api", "title": "Batman (Superhero API #70)", "url": null, "collection": null}],
  "tools_used": ["get_superhero"],
  "warnings": []
}
```

Interactive docs at http://localhost:8000/docs.

## Tests and benchmark

```bash
uv run pytest                                  # unit tests, no keys or network needed
uv run python -m benchmark.validate            # checks every benchmark answer exists in the data
uv run python -m benchmark.run                 # 125 live questions, report in benchmark/results/
```

125 questions: mythology, superhero, "both", out-of-scope, edge cases and tricky ones (known names with off-topic
requests, false premises, facts the model knows but the data doesn't, made-up names, prompt injection). An optional
LLM judge (`JUDGE_*` settings) grades behaviour keywords can't. The report breaks results down by expected behaviour,
routing, citations, BM25 retrieval quality and token cost.

### Results

Full run of the final code: 125 questions, bot `openai/gpt-oss-120b` on Groq (no fallback), judge `qwen3.8-27b`.
Report: `benchmark/results/20261023-214125-summary.md`.

| Behaviour | Result |
|---|---|
| Overall | 116/125 (93%) |
| Declines off-topic requests | 10/10 |
| Corrects false or leading premises | 5/5 |
| Resists prompt injection | 2/2 |
| Refuses to answer from outside knowledge | 4/4 |
| Says it doesn't know when the data has no answer | 10/13 |
| Answers from documents / superhero data / both | 41/44 · 28/29 · 16/18 |
| BM25: recall@3 on the raw question / hit rate with the model's own queries | 90% / 100% |
| Prompt tokens per question (24% served from Groq's cache) | ~2,100 |
| Median latency | 2.6 s |

Of the 9 failures, 2 are keyword-grader misses that the LLM judge passed, 2 cite a valid article other than the
expected one, and 1 was a Superhero API outage (reported in `warnings`). The other 4 are real limits: a spelling
variant BM25 can't match ("Balder" vs "Baldr"), a character named in the question but not looked up (Wonder Woman),
a two-figure question searched for only one figure, and a bare "I don't have that information" instead of saying
what wasn't found.

## Design in brief

- **Routing**: the LLM gets two tools, `search_documents` and `get_superhero`, in a hand-written loop (`app/agent.py`).
  When a name it asked one source about also exists in the other (Loki the god and the Marvel character), the tool
  result says so and the model decides whether to check both. No list of names is hardcoded.
- **Documents**: every folder under `data/` with a `collection.json` (a one-line description) is a collection. Adding
  a dataset needs no code change. BM25 over section-level chunks, behind a `Retriever` interface for embeddings later.
- **Sources**: tool results are labelled `[S1]`, `[S2]`…; only ids the server issued reach the response, so the model
  can't invent a source. Paragraphs left uncited get one repair turn, and any still uncited are flagged in
  `warnings`. Answers given without tools are attributed to `llm`.
- **Resilience**: an optional fallback LLM. On a rate limit or outage the whole question restarts there, because one
  provider's tool-call history is not valid input for another.

## Shortcuts taken

- No conversation memory, auth or rate limiting; cache and index are in memory.
- The Superhero API only searches by name, so "who is the strongest" questions can't be answered.
- Free tiers limit throughput: about 75 questions per day per Groq model (a question is ~2,500 tokens), and ~20
  requests per day on Gemini's free tier. A second Groq model makes the most useful free fallback.

## Data

`data/mythology/` holds Wikipedia articles (CC BY-SA 4.0), fetched with `scripts/fetch_mythology.py`.
To add a collection, create `data/<name>/` with markdown articles in the same format and a `collection.json`.
