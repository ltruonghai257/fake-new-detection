---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: milestone
status: executing
stopped_at: Phase 5 plan 05-02 complete — milestone v3.1 all phases done
last_updated: '2026-08-22T10:15:00Z'
last_activity: 2026-08-22 -- Phase 05 plan 05-02 executed
progress:
    total_phases: 3
    completed_phases: 3
    total_plans: 5
    completed_plans: 5
    percent: 100
current_phase: 5
current_phase_name: Demo App + Tests
---

# Project State

## Current Position

Phase: 05 (demo-app-tests) — COMPLETE
Plan: 2 of 2 (all plans done)
Status: Phase 05 complete — all 3 phases of v3.1 complete
Last activity: 2026-08-22 -- Phase 05 plan 05-02 executed

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
**Stopped at:** Phase 5 plan 05-02 complete — all v3.1 plans done
**Resume file:** (none — milestone complete)

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
-   [Phase 05]: Integration tests use @pytest.mark.integration and are excluded from default pytest run — prevents CI flakiness and API key leakage
-   [Phase 05]: evaluate_agent HTTP test skips when VIFACTCHECK_CKPT_DIR unset — avoids heavy model loading in environments without checkpoints
-   [Phase 05]: uuid4 per invocation in run_fact_check() and mcp_server.py — prevents MemorySaver state bleed across concurrent calls (D-10)
