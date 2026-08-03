# Roadmap: factcheck_agents v3.0 Debate-Based Verification Pipeline and Demo App

**Milestone:** v3.0 · **Status:** Active

## Overview

Replace the single-pass evidence-overrides-models correctness bug with a full debate-based multi-agent architecture — dual-source evidence retrieval, BM25+embedding reranking, conditional social-search loop, agreement gate, bounded advocate debate with JSONL audit logging, and a weighted judge — then ship a local thesis defense demo web app (FastAPI + React/Vite/TypeScript) that streams the debate live turn-by-turn via SSE with a Vietnamese UI and a verdict card showing the 30/30/40 weight breakdown.

## Phases

-   [x] **Phase 1: Debate Pipeline** - Build all debate infrastructure: dual-source agents, reranker, social loop, agreement gate, bounded debate, weighted judge, graph wiring, and tests
-   [x] **Phase 2: Demo App** - FastAPI SSE backend + React/Vite/TypeScript frontend with live debate streaming and Vietnamese verdict card

## Phase Details

### Phase 1: Debate Pipeline

**Goal**: Implement the complete debate-based verification pipeline — from dual-source evidence retrieval through weighted judge — with JSONL/JSON audit logging and full unit + integration test coverage.
**Requirements**: EVRET-01, EVRET-02, EVRET-03, EVRET-04, RERANK-01, RERANK-02, SOCLOOP-01, SOCLOOP-02, SOCLOOP-03, AGREE-01, AGREE-02, AGREE-03, DEBATE-01, DEBATE-02, DEBATE-03, JUDGE-01, JUDGE-02, JUDGE-03
**Depends on**: Nothing (first phase)
**Success Criteria** (what must be TRUE):

1. `state.py` adds all M2 fields (`evidence_real`, `evidence_fake`, `social_loop_fired`, `request_id`, `agreement_score`, `debate_turns`, `debate_exit_reason`, `weight_breakdown`) with `total=False`; all existing tests continue to pass unchanged
2. `reranker.py` BM25 + embedding rerank selects top-k evidence snippets within PhoBERT's 256-token budget; RERANK-02 unit test passes recall@k on a labeled sample of ≥ 5 claim–snippet pairs
3. Social loop fires at most once per request; SOCLOOP-03 unit test asserts that when `social_loop_fired=True` is present on the input state, the routing function short-circuits to `verify` and never reaches `social_loop_node` again
4. `agreement_gate.py` computes `0.30 × phobert_confidence + 0.30 × coolant_confidence + 0.40 × evidence_credibility` and routes directly to judge when score ≥ `FACTCHECK_AGREEMENT_THRESHOLD`, logging `debate_exit_reason = "skipped_high_agreement"` to state and a `{"debate_skipped": true, ...}` line to `logs/debates/<request_id>.jsonl`
5. `debate_node.py` runs a bounded real/fake advocate debate (default `max_debate_rounds=2`); every turn (agent, round, ISO timestamp, full text) is printed to stdout **and** appended atomically to `logs/debates/<request_id>.jsonl` with `ensure_ascii=False`
6. `judge_agent.py` scores each argument on three 1–5 dimensions and produces `{verdict, confidence, explanation, weight_breakdown}`; full breakdown written atomically to `logs/verdicts/<request_id>.json`
7. `graph.py` exposes `build_debate_graph()` for M2 callers and keeps `build_graph()` for M1 backward compat; integration tests pass end-to-end on 2 sample Vietnamese claims

Plans:

-   [x] 01-01: Wave 1 — Extend `state.py` with all M2 `TypedDict` fields (`total=False`) and update `initial_state()` defaults; add new env vars to `config.py` (`FACTCHECK_AGREEMENT_THRESHOLD`, `FACTCHECK_MAX_DEBATE_ROUNDS`, `GOOGLE_FACTCHECK_API_KEY`, social loop thresholds)
-   [x] 01-02: Wave 2 — Implement `reranker.py` (BM25 + embedding rerank, top-k selection within 256-token budget); implement `agents/real_source_agent.py` (credible Vietnamese outlets → `evidence_real`) and `agents/fake_source_agent.py` (tingia.gov.vn + Google Fact Check API stub → `evidence_fake`)
-   [x] 01-03: Wave 3 — Implement `agents/social_loop_agent.py` (one-shot weak-evidence social search, sets `social_loop_fired=True`) and `agents/agreement_gate.py` (0.30/0.30/0.40 formula, evidence-credibility sub-components, routing logic); write unit tests for reranker recall@k and social loop fire-once guard
-   [x] 01-04: Wave 4 — Implement `agents/debate_node.py` (bounded Python `for` loop advocate debate, atomic JSONL logging to `logs/debates/`) and `agents/judge_agent.py` (1–5 dimension scoring, weighted verdict, atomic JSON logging to `logs/verdicts/`); update `graph.py` with full M2 topology and `build_debate_graph()` function
-   [x] 01-05: Wave 5 — Integration tests: run full M2 pipeline on 2 sample Vietnamese claims end-to-end; assert verdict structure, log files created, no test regressions against existing 83 tests

---

### Phase 2: Demo App

**Goal**: Ship a local-only thesis defense demo web app with a FastAPI SSE backend and a React/Vite/TypeScript frontend that streams the debate live and displays a Vietnamese verdict card.
**Requirements**: DEMO-01, DEMO-02, DEMO-03, DEMO-04
**Depends on**: Phase 1 (stable `run_fact_check()` programmatic entry point with SSE-compatible streaming hooks)
**Success Criteria** (what must be TRUE):

1. `demo_app/backend/main.py` starts with `uvicorn`; `POST /api/analyze` accepts `{statement: str, image_path?: str}` and initiates the debate pipeline via direct Python import (no subprocess)
2. SSE stream emits all required event types — `stage_start`, `turn_start`, `chunk`, `turn_end`, `verdict`, `heartbeat` (every 5 s) — and client disconnect is detected and aborts the pipeline loop
3. Debate stage streams character-level chunks (~8 chars / 20 ms) in real time; React frontend renders alternating chat bubbles (blue = `real_advocate`, red = `fake_advocate`) with argument quality score badges per turn
4. Final verdict card shows label, confidence gauge, 30/30/40 weight breakdown bar, and working download buttons for `logs/debates/<id>.jsonl` and `logs/verdicts/<id>.json`
5. All UI copy is in Vietnamese; CORS allows only `http://localhost:5173`; `useEffect` SSE cleanup closes `EventSource` on unmount (React StrictMode safe); no auth, no public deployment path

Plans:

-   [x] 02-01: FastAPI backend — `demo_app/backend/main.py` (app, CORS, `/api/analyze` POST + SSE GET), `demo_app/backend/streaming.py` (SSE generator + `asyncio.Queue` bridge to pipeline), heartbeat task, client disconnect handling
-   [x] 02-02: React/Vite/TypeScript frontend — scaffold `demo_app/frontend/` with Vite + Tailwind; implement `App.tsx`, `DebateTranscript.tsx` (alternating bubbles, score badges), `VerdictCard.tsx` (label, confidence, weight bar, log downloads), `EvidencePanel.tsx` (tier badges); native `EventSource` SSE client with StrictMode-safe cleanup

---

## Dependency Map

```
Phase 1 (Debate Pipeline) ──► Phase 2 (Demo App)
```

Phase 1 must be complete and `build_debate_graph()` stable before Phase 2 begins.

## Progress

| Phase | Name            | Plans | Status   |
| ----- | --------------- | ----- | -------- |
| 1     | Debate Pipeline | 5/5   | Complete |
| 2     | Demo App        | 2/2   | Complete ||

_Created: 2026-08-02_
