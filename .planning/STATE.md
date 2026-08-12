---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: A2A Protocol Integration
status: planning
last_updated: '2026-08-13T00:00:00.000Z'
last_activity: 2026-08-13 — Milestone v3.1 started; v3.0 complete (Phase 1 + Phase 2 shipped)
progress:
    total_phases: 3
    completed_phases: 0
    total_plans: 0
    completed_plans: 0
    percent: 0
current_phase: 1
---

# Project State

## Current Position

Phase: Phase 1 (A2A Agent Wrappers)
Plan: —
Status: Planning
Last activity: 2026-08-13 — Milestone v3.1 started; v3.0 complete (Phase 1 + Phase 2 shipped)

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
