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

**Model: `openai/gpt-oss-120b` on Groq**, no fallback, on the final code (125 questions).
Every answer was graded by Claude (Anthropic), reading it against the question and the retrieved sources
(`benchmark/results/20261023-214125-review.md`). Grading is a benchmark tool only; it is not part of the
production path.

| Behaviour | Result |
|---|---|
| Overall | **122/125 (98%)** |
| Declines off-topic requests | 10/10 |
| Corrects false or leading premises | 5/5 |
| Resists prompt injection | 2/2 |
| Refuses to answer from outside knowledge | 4/4 |
| Says it doesn't know when the data has no answer | 13/13 |
| Answers from documents / superhero data / both | 43/44 · 29/29 · 16/18 |
| BM25 (4 passages per search): recall on the raw question / with the model's own queries | 95% / 100% |
| Prompt tokens per question (24% served from Groq's cache) | ~2,100 |
| Median latency | 2.5 s |

The three incorrect answers:

- "Who killed Balder?": retrieval missed "Höðr" (the tokenizer leaves ð unfolded) and the spelling "Balder".
- Wonder Woman's mother: right answer, but the Wonder Woman link came from the model, not a lookup.
- Thor vs Hercules: a two-figure question searched for only one figure.

## Design in brief

- **Routing**: the LLM gets two tools, `search_documents` and `get_superhero`, in a hand-written loop (`app/agent.py`).
  When a name it asked one source about also exists in the other (Loki the god and the Marvel character), the tool
  result says so and the model decides whether to check both. No list of names is hardcoded.
- **Documents**: every folder under `data/` with a `collection.json` (a one-line description) is a collection. Adding
  a dataset needs no code change. BM25 over section-level chunks, behind a `Retriever` interface for embeddings later.
- **Sources**: tool results are labelled `[S1]`, `[S2]`…; only ids the server issued reach the response, so the model
  can't invent a source. Uncited claims get one repair turn, and any still uncited are flagged in `warnings`.
  Answers given without tools are attributed to `llm`.
- **Resilience**: an optional fallback LLM. On a rate limit or outage the whole question restarts there, because one
  provider's tool-call history is not valid input for another.

## Shortcuts taken

- No conversation memory, auth or rate limiting; cache and index are in memory.
- The Superhero API only searches by name, so "who is the strongest" questions can't be answered.
- Free tiers limit throughput: a question costs ~2,400 tokens including the answer, so Groq's 200k tokens per day
  is about 80 questions per model, and Gemini's free tier allows ~20 requests per day. A second Groq model makes
  the most useful free fallback.

## Data

`data/mythology/` holds Wikipedia articles (CC BY-SA 4.0), fetched with `scripts/fetch_mythology.py`.
To add a collection, create `data/<name>/` with a `collection.json` such as `{"description": "Wikipedia articles on
Egyptian mythology"}` (shown to the model) and one markdown file per article: a `# Title` line, a
`Source: <url>` line, then `## Section` headings with paragraphs. Restart the app; no code changes are needed.
