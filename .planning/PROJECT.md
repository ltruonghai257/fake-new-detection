# fake-new-detection / factcheck_agents

## What This Is

A multi-agent LangGraph module (`factcheck_agents/`) that fact-checks Vietnamese news statements
using web evidence and two trained models (PhoBERT ViFactCheck + COOLANT). Agents share a single
typed state object and run in sequence — inspired by TradingAgents' shared-state design.
The module is fully decoupled from the training pipeline; model checkpoints are lazy-loaded and
every failure degrades gracefully.

## Core Value

A user submits a Vietnamese claim and gets back a binary verdict — **Thật** or **Giả** — with a
Vietnamese-language rationale and citations they can verify, even when model checkpoints are missing.

## Current Milestone: v3.0 Debate-Based Verification Pipeline and Demo App

**Goal:** Replace the evidence-overrides-models correctness bug with a debate-based multi-agent
architecture and ship a local thesis defense demo web app with live streaming.

**Target features:**

-   Dual-source evidence retrieval: `real_source_agent` (credible Vietnamese outlets) + `fake_source_agent` (tingia.gov.vn + Google Fact Check Tools API stub); outputs as separate typed lists `evidence_real` / `evidence_fake`; both empty → NEI gate
-   M-ReRank evidence reranking: BM25 + embedding rerank (or cross-encoder) of all snippets before PhoBERT 256-token truncation; top-k selected for maximum recall; unit-tested on labeled sample
-   Conditional social-search loop: one-shot extra social search (TikTok FactCheckVN + flagged pages) when evidence non-empty but weak (count + credibility below concrete thresholds); hard-capped at 1 execution per request
-   Agreement gate: compute agreement score across PhoBERT, COOLANT, and evidence signals; skip debate at ≥ 0.8 threshold (configurable), log "debate_skipped: high agreement"
-   Bounded debate loop: `real_advocate` + `fake_advocate` LLM agents, rebuttal-based, default `max_debate_rounds=2` (configurable); every turn logged to stdout AND `logs/debates/<request_id>.jsonl`
-   Weighted judge: argument quality scoring (1-5) + PhoBERT 30% + COOLANT 30% + evidence-credibility 40% (tier + count + directional consistency); `{verdict, confidence, explanation, weight_breakdown}`; full breakdown to `logs/verdicts/<request_id>.json`
-   Demo web app: `demo_app/` — FastAPI + React/Vite/TypeScript; SSE debate streaming turn-by-turn; Vietnamese UI; verdict card with weight bar + log download; local-only

## Requirements

### Validated

<!-- Shipped v1.0 baseline — confirmed working -->

-   ✓ **CLI-01**: User can fact-check a statement via `python -m factcheck_agents.cli` — v1.0
-   ✓ **CLI-02**: User can supply an optional image path to enable COOLANT — v1.0
-   ✓ **API-01**: `run_fact_check(statement)` Python API returns verdict + rationale + citations — v1.0
-   ✓ **MCP-01**: MCP server exposes `fact_check`, `search_evidence`, `evaluate_statement` tools — v1.0
-   ✓ **SEARCH-01**: Search agent drafts queries (LLM or heuristic fallback) and retrieves web evidence via Tavily/Google CSE — v1.0
-   ✓ **EVAL-01**: Evaluate agent runs PhoBERT ViFactCheck (statement + evidence → SUPPORTED/REFUTED/NEI) — v1.0
-   ✓ **EVAL-02**: Evaluate agent runs COOLANT (image → REAL/FAKE) only when image supplied — v1.0
-   ✓ **EVAL-03**: Missing model checkpoints produce `unavailable` result; pipeline never crashes — v1.0
-   ✓ **CONCL-01**: Conclusion agent fuses model verdicts + evidence into 4-class verdict (TRUE/FALSE/MISLEADING/UNVERIFIED) — v1.0
-   ✓ **CONCL-02**: Rule-based fallback verdict when LLM is unavailable — v1.0
-   ✓ **CFG-01**: All settings come from env vars (no hardcoded secrets) — v1.0

### Active

<!-- v3.0 scope — M2: Debate-Based Verification Pipeline and Demo App -->

-   [ ] **EVRET-01**: `real_source_agent` searches credible Vietnamese outlets only (vnexpress.net, tuoitre.vn, thanhnien.vn, ttxvn.gov.vn, vtv.vn, dantri.com.vn)
-   [ ] **EVRET-02**: `fake_source_agent` searches tingia.gov.vn and Google Fact Check Tools API (stubbed without API key; gated on user providing key)
-   [ ] **EVRET-03**: Outputs stay as separate typed lists (`evidence_real`, `evidence_fake`) — never merged into a boolean
-   [ ] **EVRET-04**: Both lists empty → NEI gate with confidence 0.0; rest of pipeline skipped
-   [ ] **AGREE-01**: Agreement score computed across PhoBERT, COOLANT, and evidence signals after parallel model run + evidence collection
-   [ ] **AGREE-02**: Agreement ≥ configurable threshold (default 0.8) → skip debate, log "debate_skipped: high agreement"
-   [ ] **DEBATE-01**: `real_advocate` and `fake_advocate` LLM agents run rebuttal-based debate (`max_debate_rounds=2`, configurable via env)
-   [ ] **DEBATE-02**: Every turn (agent, round, timestamp, full text) printed to stdout AND appended to `logs/debates/<request_id>.jsonl`
-   [ ] **JUDGE-01**: Judge scores each argument 1-5 (factuality + engagement with rebuttal) and combines PhoBERT 30% + COOLANT 30% + evidence-credibility 40%
-   [ ] **JUDGE-02**: Evidence-credibility computed from source tier + count + directional consistency (never a binary flag)
-   [ ] **JUDGE-03**: Output `{verdict: Real|Fake|NEI, confidence, explanation, weight_breakdown}`; full breakdown to `logs/verdicts/<request_id>.json`
-   [ ] **RERANK-01**: Before truncating evidence to PhoBERT's 256-token limit, rerank all snippets from `evidence_real` + `evidence_fake` by relevance to the claim using BM25 + embedding rerank (or cross-encoder if available in repo); select top-k that maximize recall within budget; unit test asserts recall@k ≥ acceptable threshold on a small labeled sample
-   [ ] **SOCLOOP-01**: Conditional edge between evidence retrieval (REQ 1) and agreement gate (REQ 2): if `evidence_real` and `evidence_fake` are both non-empty but weak (count < threshold AND/OR credibility score < threshold — concrete values defined in implementation), trigger exactly ONE additional social-media search round (TikTok FactCheckVN, previously flagged pages); hard cap: loop never fires more than once per request; unit test asserts second execution is blocked
-   [ ] **DEMO-01**: `demo_app/` — FastAPI backend + React/Vite/TypeScript frontend; POST `/analyze` with SSE streaming
-   [ ] **DEMO-02**: Debate stage streams turn-by-turn live as alternating advocate chat bubbles; Vietnamese UI; local-only

### Out of Scope

-   Modifying PhoBERT/COOLANT model weights, architecture, or training code
-   Removing the existing source-tier system (extend it for evidence-credibility in JUDGE-02)
-   Public deployment of the demo app
-   Paid X/Twitter API or Meta Graph API
-   Modifying `training/`, model checkpoints, or notebooks

## Context

-   **Stack**: Python, LangGraph, PhoBERT (vinai/phobert-base-v2 via HuggingFace), COOLANT (custom checkpoint), Tavily/Google CSE, FastMCP, FastAPI, React/Vite/TypeScript (demo_app/)
-   **Architecture reference**: TradingAgents (shared state, Bull/Bear debate → judge), ViFactCheck AAAI 2025, Debate-to-Detect / TruEDebate / DebateCV (multi-dimensional argument scoring)
-   **Evidence graph pattern**: plain Python networkx structure, built once, queried downstream
-   **Binary mapping default**: MISLEADING → FAKE, UNVERIFIED → FAKE ("not verifiably real" cannot be reported as Thật)
-   **Debate weights**: PhoBERT 30% + COOLANT 30% + evidence-credibility 40% — flagged as tunable defaults
-   **Graceful degrade**: a missing checkpoint marks that sub-verdict `unavailable`; always preserved

## Constraints

-   **Scope**: Work only inside `factcheck_agents/` and `tests/` — never touch `training/` or notebooks
-   **Dependencies**: No new paid APIs without explicit confirmation; prefer packages already in `requirements.txt`
-   **Compatibility**: New binary verdict fields must be additive (no breaking change to existing callers)
-   **Language**: All user-facing output in Vietnamese; internal field/enum names stay English for API stability
-   **Evidence graph lib**: `networkx` is acceptable; also a plain `dict`-of-dicts is sufficient — keep it minimal

## Key Decisions

| Decision                                                 | Rationale                                                                                   | Outcome        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------- |
| LangGraph for orchestration                              | Shared-state, conditional edges, composable nodes                                           | ✓ Good         |
| Lazy model loading with `lru_cache`                      | Avoids import-time torch overhead; degrades when checkpoint missing                         | ✓ Good         |
| `unavailable` result (not raise) for missing checkpoints | Pipeline must always complete; callers check `available` field                              | ✓ Good         |
| Tavily primary / Google CSE fallback                     | Tavily returns cleaner LLM-ready snippets; CSE as resilience fallback                       | ✓ Good         |
| 4-class verdict → 2-class binary (v2.0)                  | MISLEADING/UNVERIFIED cannot be reported as Thật; binary is user-facing                     | ✓ Shipped v2.0 |
| Evidence graph as plain Python structure (v2.0)          | Avoid new heavy deps; "build once" pattern without npm                                      | ✓ Shipped v2.0 |
| Debate architecture over single-pass verdict (v3.0)      | Single-pass evidence silently overrides model predictions; debate forces explicit weighting | — Pending      |
| 30/30/40 weight split (v3.0)                             | Equal model weight, evidence-credibility plurality; tunable default in STATE.md             | — Pending      |
| Google Fact Check API stubbed until key provided (v3.0)  | Avoid silent failures; user must opt in explicitly                                          | — Pending      |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):

1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):

1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---

_Last updated: 2026-08-02 — Milestone v3.0 started (M2: Debate-Based Verification Pipeline and Demo App)_
