# Week 3 Project — Layoffs Research Agent

**Mastering Agentic AI Bootcamp | The Gen Academy | Week 3: Build Your AI Agent**

## The One-Liner

My agent helps **a researcher or journalist tracking tech layoffs** investigate **a specific
company's layoff history** in **a Streamlit chat surface**, replacing **manually cross-referencing
the Week 2 corpus with a live news search whenever it doesn't cover a company**. It checks the
indexed corpus first, and if coverage is thin, autonomously **searches the web, scrapes the top
article, and drafts a cited answer** on its own using **4 tools** (corpus retrieval, web search,
article scraping, structured fact extraction), **hands off to a human before the newly researched
article is permanently added to the corpus** (the one write action in the system), and I'll know it
works when a user gets an accurate, cited answer for both in-corpus and previously-uncovered
companies, with every corpus write approved by a person first.

**Track:** Option 2 (bring your own use case) + Track 2 (LangChain/LangGraph).

## Project Overview

Week 2 built a RAG app over a static, 60-article layoffs-news corpus: good for answering
questions about companies already indexed, but it had no way to handle a company it didn't
know about — it just refused. Week 3 turns that static RAG app into an agent: a LangGraph
state machine that decides what to do next instead of doing one retrieval pass and stopping.

If the corpus already covers the question, the agent answers from it immediately — cheap and
fast, no reason to touch the network. If it doesn't, the agent autonomously calls out to the web:
search → scrape the most promising result → extract structured facts and draft a cited answer.
That's the autonomous part. The write — permanently adding the new article to the corpus so
future questions benefit from it — always pauses for a human to approve first, via LangGraph's
`interrupt()`. This is the literal embodiment of the handout's "write actions deserve a human"
rule: reads (retrieve, search, scrape) run on their own; the one action that mutates persistent
state does not.

## Architecture

```
retrieve_corpus --score >= 0.62--> answer_from_corpus --> END
      |
  score < 0.62
      v
  web_search --no hits--> handle_error --> END
      |
  hits found
      v
scrape_article --all scrape attempts failed--> handle_error --> END
      |
  scraped ok
      v
extract_and_draft (LLM: structured facts + cited answer from the one fresh article)
      |
      v
human_approval  <-- LangGraph interrupt(); Streamlit renders Approve/Reject
      |
 +----+----+
 |         |
approved  rejected
 |         |
save_to_corpus   (answer still shown, nothing persisted)
 |
END
```

Built with LangGraph's `StateGraph` + `InMemorySaver` checkpointer (session-scoped; a fresh
`thread_id` per question). State carries the corpus index (embeddings + metadata) through the
whole graph, including across the interrupt/resume boundary — confirmed this survives fine
since the checkpointer holds live Python objects in-process rather than serializing to JSON.

## What's reused from Week 2 vs. new this week

| Component | Status |
|---|---|
| `corpus/articles.jsonl` (60 articles) | Copied from Week 2 as the starting corpus |
| Chunking + Nebius embeddings + hybrid dense/BM25 retrieval | Reused from Week 2's `rag.py`, now in `retrieval.py` |
| Corpus-covered answer generation | Reused from Week 2 |
| **Agentic control flow (LangGraph graph, routing, retries)** | **New** |
| **Web search + scrape + structured extraction tools** | **New** (`tools.py`) |
| **Human-in-the-loop approval before any corpus write** | **New** (`agent.py`, `interrupt()`) |
| **Incremental corpus growth** (`add_article_to_index`) | **New** — corpus grows across sessions as questions get approved, instead of being a fixed one-time snapshot |
| **Error handling for search/scrape failure** | **New** (`handle_error` node) |

## Human-in-the-Loop / Hard Limits

- **Never** writes to the corpus without an explicit human click. `route_after_approval` only
  reaches `save_to_corpus` when `interrupt()`'s resume payload has `approve: true`; rejecting
  still returns the drafted answer, just without persisting it.
- **Never** fabricates a citation: the corpus-answer prompt and the web-answer prompt both
  explicitly instruct "use ONLY the provided source(s)," matching Week 2's refusal discipline.
- **What happens when something breaks:** `web_search` and `scrape_article` swallow their own
  exceptions and return `[]`/`None` rather than raising; the graph routes any of those failure
  states to `handle_error`, which returns an honest "couldn't find this" message instead of a
  stack trace. Verified by mocking both failure points (empty search, failed scrape) — both
  routed to `handle_error` correctly without crashing the graph.

## Iterations

1. **Chat model degraded, not migrated on purpose.** Week 2's `meta-llama/Llama-3.3-70B-Instruct`
   now 403s on the Nebius account — the catalog has moved on. Queried `/v1/models` again (same
   move as Week 2's embedding-model iteration) and switched the default to
   `Qwen/Qwen3-235B-A22B-Instruct-2507`.
2. **The corpus/web routing threshold needed recalibrating, not just reusing.** Carried over
   Week 2's `MIN_COMBINED_SCORE = 0.58` threshold verbatim at first, and it broke: a query about
   a company with zero corpus coverage ("What happened with layoffs at Intel in 2025?") still
   routed to the corpus branch. Root-caused it to two BM25 bugs, not just a bad threshold:
   - Tokenizing with a bare `.split()` never stripped punctuation, so `"Capiter?"` from a query
     never matched the corpus's `"capiter"` token at all — the *true positive* case was
     actually scoring artificially low.
   - With no stopword removal, function words ("what", "at", "in") and the domain-wide word
     "layoffs" (present in nearly every chunk, since every article in this corpus is about a
     layoff) dominated the BM25 score, letting an unrelated chunk (Rivian's boilerplate) outscore
     the real answer for an Intel query purely on incidental word overlap.
   Fixed both (regex word tokenizer + stopword list including domain-generic terms), then
   re-ran the same calibration method Week 2 used — sampled 3 answerable and 3 unanswerable
   queries — and moved the threshold to 0.62, which now cleanly separates all six.
3. **Verified the interrupt/resume boundary explicitly before trusting it.** LangGraph's
   `interrupt()` pauses a node and needs `Command(resume=...)` to continue; before wiring this
   into Streamlit I isolated it in a standalone script to confirm (a) the exact shape of
   `result["__interrupt__"]`, and (b) that a numpy array sitting in state (the embeddings index)
   survives the pause/resume round-trip intact, since `InMemorySaver` keeps live objects rather
   than serializing them to JSON.
4. **A false alarm during manual testing, traced with debug prints instead of guessed at.** An
   early browser test appeared to auto-approve a corpus write with no button click. Rather than
   patch around it blind, added temporary print-tracing around both `graph.invoke()` call sites
   and reran with an explicit send-button click instead of a synthetic Enter keypress — the
   approval step paused exactly as expected. The original anomaly was an artifact of the
   browser-automation harness (an Enter keystroke racing a page rerender), not a bug in the
   agent or the app; removed the tracing once confirmed.

## Learnings / Observations

- **"Insufficient corpus coverage" is a harder detector to build than it looks, and it's the
  hinge the whole agent turns on.** Get it wrong in one direction and the agent silently refuses
  to research things it should; get it wrong the other way and it answers from the wrong source
  material entirely. It deserves the same treatment as Week 2's refusal threshold: sampled,
  measured, and recalibrated after a fix — not something to eyeball once and move on from.
- **Read/write separation is what makes an agent that could otherwise be dangerous instead feel
  safe.** Everything on the "insufficient coverage" branch — searching, scraping, drafting — runs
  fully autonomously with no human in the loop, because none of it can do harm on its own. The
  one step that permanently changes shared state (the corpus every future user's questions will
  be answered from) is the one step gated on a person clicking a button. That split, not any
  prompt, is what makes the system trustworthy.
- **State surviving an interrupt is not something to assume — check it.** A LangGraph graph
  checkpoints its *entire* state object across a pause, including things that aren't obviously
  "state" in the conversational sense (here, a numpy embeddings matrix). It happened to work
  because `InMemorySaver` doesn't serialize, but that's a property of the checkpointer choice,
  not something guaranteed by the interrupt API in general — worth remembering before swapping
  in a persistent checkpointer (e.g., SQLite) for a real deployment, since that would force the
  state to actually serialize.
- **Building this on top of Week 2, rather than as a new project, was itself informative.**
  Every piece Week 2 got right (hybrid retrieval, calibrated refusal, cited generation) could be
  reused directly; every piece Week 2 didn't need to get right yet (routing decisions under
  ambiguity, tool failure, a persistent write) is exactly what became the actual Week 3 work.
  The corpus even keeps growing from approved research — the two weeks now share one asset.

## Repository

Code: https://github.com/anirudhV90/GenAI-Week3 *(update after push)*
