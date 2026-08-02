---
gsd_state_version: 1.0
milestone: v3.0
milestone_name: milestone
status: Executing
current_phase: 2
last_updated: '2026-08-03T00:00:00.000Z'
last_activity: 2026-08-03 — Phase 1 (Debate Pipeline) execution complete; all 5 waves executed successfully
progress:
    total_phases: 2
    completed_phases: 1
    total_plans: 5
    completed_plans: 5
    percent: 50
---

# Project State

## Current Position

Phase: Phase 2 (Demo App)
Plan: —
Status: Executing
Last activity: 2026-08-03 — Phase 1 (Debate Pipeline) execution complete; all 5 waves executed successfully

## Milestone v3.0 Requirements Index

| REQ-ID        | Category           | Scope   | Description                                            |
| ------------- | ------------------ | ------- | ------------------------------------------------------ |
| EVRET-01..04  | Evidence Retrieval | Phase 1 | Dual-source agents (real + fake), NEI gate             |
| RERANK-01     | Evidence Reranking | Phase 1 | BM25 + embedding rerank before PhoBERT truncation      |
| SOCLOOP-01    | Social Loop        | Phase 1 | One-shot weak-evidence social search, hard-capped at 1 |
| AGREE-01..02  | Agreement Gate     | Phase 1 | Skip-debate threshold, high-agreement logging          |
| DEBATE-01..02 | Debate Loop        | Phase 1 | Bounded advocate debate, JSONL turn logging            |
| JUDGE-01..03  | Weighted Judge     | Phase 1 | 30/30/40 weights, structured output, verdict JSON      |
| DEMO-01..02   | Demo App           | Phase 2 | FastAPI + React/Vite/TS, SSE streaming, Vietnamese UI  |

## Tunable Defaults (flag before changing)

-   `FACTCHECK_AGREEMENT_THRESHOLD` = 0.8
-   Debate weights: PhoBERT 30% / COOLANT 30% / evidence-credibility 40%
-   `max_debate_rounds` = 2
-   Social loop weakness threshold: TBD in Phase 1 discuss (define concrete values, not vague)
