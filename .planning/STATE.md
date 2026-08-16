---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: A2A Protocol Integration
current_phase: 4
current_phase_name: LangGraph → A2A Client Wiring
status: ready
stopped_at: Phase 4 planned
last_updated: '2026-08-16T17:00:00.000Z'
last_activity: 2026-08-16
last_activity_desc: Phase 4 planned (04-01, 04-02) — ready for execution
progress:
    total_phases: 3
    completed_phases: 1
    total_plans: 1
    completed_plans: 1
    percent: 33
---

# Project State

## Current Position

Phase: 4 — LangGraph → A2A Client Wiring
Plan: 2/2 plans created (04-01, 04-02)
Status: Planned — ready for execution
Last activity: 2026-08-16 — Phase 4 planned (research + 2 plans)

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

**Last session:** 2026-08-15T19:06:28.609Z
**Stopped at:** Phase 4 context gathered
**Resume file:** .planning/phases/04-langgraph-a2a-client-wiring/04-CONTEXT.md
