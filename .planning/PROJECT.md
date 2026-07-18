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

## Current Milestone: v2.0 Evidence-Graph Vietnamese Pipeline

**Goal:** Extend the 3-node pipeline into a 3-stage evidence-graph-driven system with binary
Thật/Giả output and fully Vietnamese user-facing text.

**Target features:**
- Evidence graph per check (in-memory, plain Python — entities, snippets, source tier, relations)
- Source tiers: env-configurable trusted domains (vnexpress, thanhnien, dantri, tuoitre) + flagged low-reliability (kenh14); tagged on every evidence hit
- Evidence/Search Agent: separate tier queries, builds graph consumable by downstream agents
- Verify Agent: PhoBERT + COOLANT in parallel → reliability signal → conditional social-media search (site-restricted Tavily/Google CSE, no paid X/Meta API) → merge into graph
- Conclusion Agent: binary verdict (`verdict: "REAL"|"FAKE"`, `verdict_label_vi: "Thật"|"Giả"`) + Vietnamese rationale + citations; 4-class signal preserved inside rationale
- Conditional graph edge: Verify → (reliable? social-search → Conclusion : Conclusion)
- CLI/API/MCP: new binary fields added non-breakingly

## Requirements

### Validated

<!-- Shipped v1.0 baseline — confirmed working -->

- ✓ **CLI-01**: User can fact-check a statement via `python -m factcheck_agents.cli` — v1.0
- ✓ **CLI-02**: User can supply an optional image path to enable COOLANT — v1.0
- ✓ **API-01**: `run_fact_check(statement)` Python API returns verdict + rationale + citations — v1.0
- ✓ **MCP-01**: MCP server exposes `fact_check`, `search_evidence`, `evaluate_statement` tools — v1.0
- ✓ **SEARCH-01**: Search agent drafts queries (LLM or heuristic fallback) and retrieves web evidence via Tavily/Google CSE — v1.0
- ✓ **EVAL-01**: Evaluate agent runs PhoBERT ViFactCheck (statement + evidence → SUPPORTED/REFUTED/NEI) — v1.0
- ✓ **EVAL-02**: Evaluate agent runs COOLANT (image → REAL/FAKE) only when image supplied — v1.0
- ✓ **EVAL-03**: Missing model checkpoints produce `unavailable` result; pipeline never crashes — v1.0
- ✓ **CONCL-01**: Conclusion agent fuses model verdicts + evidence into 4-class verdict (TRUE/FALSE/MISLEADING/UNVERIFIED) — v1.0
- ✓ **CONCL-02**: Rule-based fallback verdict when LLM is unavailable — v1.0
- ✓ **CFG-01**: All settings come from env vars (no hardcoded secrets) — v1.0

### Active

<!-- v2.0 scope — building toward these -->

- [ ] **EVGRAPH-01**: System builds an in-memory evidence graph (entities, snippets, source tier, relations) from search results, consumable by downstream agents without re-fetching
- [ ] **EVGRAPH-02**: Evidence items are tagged with source tier (`trusted` / `flagged` / `unknown`) using env-configurable domain lists
- [ ] **EVGRAPH-03**: Trusted domains (`FACTCHECK_TRUSTED_DOMAINS`) and flagged domains (`FACTCHECK_FLAGGED_DOMAINS`) are comma-separated env vars with sensible defaults
- [ ] **VERIFY-01**: Verify Agent runs PhoBERT ViFactCheck and COOLANT concurrently (where checkpoints exist) against evidence-graph context
- [ ] **VERIFY-02**: Verify Agent computes a `reliability_signal` (bool) fusing both model outputs
- [ ] **VERIFY-03**: Social-media evidence pass (X/Facebook via site-restricted search) runs only when `reliability_signal` is positive; results merge into the evidence graph
- [ ] **VERIFY-04**: Social-media search uses existing Tavily/Google CSE tools — no paid X/Meta API dependency
- [ ] **CONCL-03**: Conclusion Agent applies "any conflict ⇒ Giả, all consistent ⇒ Thật" rule to produce a binary verdict
- [ ] **CONCL-04**: Conclusion Agent outputs `verdict: "REAL"|"FAKE"` and `verdict_label_vi: "Thật"|"Giả"` fields in the verdict dict
- [ ] **CONCL-05**: All user-facing text (verdict label, rationale, evidence commentary) is written in Vietnamese
- [ ] **CONCL-06**: Internal 4-class signal (TRUE/FALSE/MISLEADING/UNVERIFIED) is preserved inside rationale for nuance, not surfaced as primary decision
- [ ] **GRAPH-01**: LangGraph graph has a conditional edge: Verify → social-search-sub-step → Conclusion (if reliable) or Verify → Conclusion (if not reliable)
- [ ] **OUTPUT-01**: CLI, Python API (`run_fact_check`), and MCP server (`fact_check`) all surface `verdict_label_vi` and binary `verdict` without breaking existing callers
- [ ] **TEST-01**: Unit tests cover evidence graph construction, source-tier tagging, reliability signal fusion, binary verdict mapping, and Vietnamese output presence

### Out of Scope

- `@colbymchenry/codegraph` npm package — it indexes source code, not text evidence; evidence graph is plain Python
- Paid X/Twitter API or Meta Graph API — only site-restricted Tavily/Google CSE; stop and ask if this changes
- Modifying `training/`, model checkpoints, or anything outside `factcheck_agents/`
- Changing the binary verdict to use more than 2 classes — MISLEADING/UNVERIFIED collapse to FAKE by design

## Context

- **Stack**: Python, LangGraph, PhoBERT (vinai/phobert-base-v2 via HuggingFace), COOLANT (custom checkpoint), Tavily/Google CSE, FastMCP
- **Architecture reference**: TradingAgents (shared state, parallel analysts → fused decision), ViFactCheck AAAI 2025 (statement + evidence → SUPPORTED/REFUTED/NEI)
- **Evidence graph pattern**: inspired by codegraph's "build once, query downstream" idea — but implemented as a plain Python structure (`networkx` or dict), not AST indexing
- **Binary mapping default**: MISLEADING → FAKE, UNVERIFIED → FAKE ("not verifiably real" cannot be reported as Thật)
- **Graceful degrade**: a missing checkpoint marks that sub-verdict `unavailable`; always preserved

## Constraints

- **Scope**: Work only inside `factcheck_agents/` and `tests/` — never touch `training/` or notebooks
- **Dependencies**: No new paid APIs without explicit confirmation; prefer packages already in `requirements.txt`
- **Compatibility**: New binary verdict fields must be additive (no breaking change to existing callers)
- **Language**: All user-facing output in Vietnamese; internal field/enum names stay English for API stability
- **Evidence graph lib**: `networkx` is acceptable; also a plain `dict`-of-dicts is sufficient — keep it minimal

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| LangGraph for orchestration | Shared-state, conditional edges, composable nodes | ✓ Good |
| Lazy model loading with `lru_cache` | Avoids import-time torch overhead; degrades when checkpoint missing | ✓ Good |
| `unavailable` result (not raise) for missing checkpoints | Pipeline must always complete; callers check `available` field | ✓ Good |
| Tavily primary / Google CSE fallback | Tavily returns cleaner LLM-ready snippets; CSE as resilience fallback | ✓ Good |
| 4-class verdict → 2-class binary (v2.0) | MISLEADING/UNVERIFIED cannot be reported as Thật; binary is user-facing | — Pending |
| Evidence graph as plain Python structure (v2.0) | Avoid new heavy deps; codegraph-style "build once" pattern without npm | — Pending |

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
*Last updated: 2026-07-19 — Milestone v2.0 started (brownfield bootstrap)*
