---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: A2A Protocol Integration
current_phase: 5
current_phase_name: Demo App + Tests
status: planned
stopped_at: Phase 5 plans approved — ready to execute
last_updated: '2026-08-22T00:00:00.000Z'
last_activity: 2026-08-22
last_activity_desc: Phase 5 plans created and verified (8/8 dimensions passed)
progress:
    total_phases: 3
    completed_phases: 2
    total_plans: 5
    completed_plans: 4
    percent: 80
---

# Project State

## Current Position

Phase: 5 — Demo App + Tests
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-18 — Phase 04 UAT complete (5/5), transitioned to Phase 5

## Milestone v3.1 Requirements Index

| REQ-ID | Category             | Scope   | Description                                                      |
| ------ | -------------------- | ------- | ---------------------------------------------------------------- |
| A2A-01 | SDK + Agent Handlers | Phase 1 | a2a-sdk installed; all 10 agents implement TaskHandler           |
| A2A-02 | Agent Cards          | Phase 1 | /.well-known/agent.json per agent for service discovery          |
| A2A-03 | Launch scripts       | Phase 1 | start_agents.sh / stop_agents.sh for local dev (ports 9001–9010) |
| A2A-04 | A2A client module    | Phase 2 | a2a_client.py wraps A2AClient; LangGraph nodes call it           |
| A2A-05 | Graph refactor       | Phase 2 | build_debate_graph() + build_graph() use A2A clients             |
| A2A-06 | Demo app SSE bridge  | Phase 3 | streaming.py calls A2A HTTP endpoints; SSE events unchanged      |
| A2A-07 | Tests updated        | Phase 3 | Agent tests via HTTP interface; graph integration tests for A2A  |
| A2A-08 | No external breakage | Phase 3 | CLI / run_fact_check() / MCP server unchanged externally         |

## Tunable Defaults (carry-over from v3.0)

-   `FACTCHECK_AGREEMENT_THRESHOLD` = 0.8
-   Debate weights: PhoBERT 30% / COOLANT 30% / evidence-credibility 40%
-   `max_debate_rounds` = 2
-   A2A agent ports: search=9001, evaluate=9002, real_source=9003, fake_source=9004,
    social_loop=9005, agreement_gate=9006, real_advocate=9007, fake_advocate=9008,
    judge=9009, conclusion=9010

## Session

**Last session:** 2026-08-22
**Stopped at:** Phase 5 plans approved — ready to execute
**Resume file:** .planning/phases/05-demo-app-tests/05-01-PLAN.md

## Performance Metrics

| Plan         | Duration | Tasks   | Files   |
| ------------ | -------- | ------- | ------- |
| Phase 04 P01 | 14       | 5 tasks | 3 files |
| Phase 04 P02 | 74       | 7 tasks | 5 files |

## Decisions

-   [Phase 04]: Sync httpx.Client bridge for A2A calls (D-01) — Sync httpx.Client bridge for A2A calls (D-01)
-   [Phase 04]: EvidenceGraph rebuilt locally in search_agent wrapper (D-03) — EvidenceGraph rebuilt locally in search_agent wrapper (D-03)
-   [Phase 04]: Patch targets stay on factcheck_agents.graph (a2a_client patches ineffective with from-import bindings) — Patch targets stay on factcheck_agents.graph (a2a_client patches ineffective with from-import bindings)
-   [Phase 04]: EvidenceGraph made checkpointer-serializable via \_asdict()/graph_data round-trip — EvidenceGraph made checkpointer-serializable via \_asdict()/graph_data round-trip
