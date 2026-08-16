---
phase: 04-langgraph-a2a-client-wiring
status: passed
verified: 2026-08-17
requirements: [A2A-04, A2A-05, A2A-05b]
---

# Phase 04 Verification: LangGraph → A2A Client Wiring

**Method:** Inline verification (verifier agent unavailable in this runtime) —
must_haves checked against the live codebase, requirement IDs cross-referenced
against REQUIREMENTS.md, and behavior validated by unit checks + the live CLI
smoke test (servers up and down).

## Requirement Traceability

| REQ-ID | Plan | Status | Evidence |
| ------ | ---- | ------ | -------- |
| A2A-04 | 04-01 | SATISFIED | `a2a_client.py` — 10 typed wrappers (`state -> dict` signatures verified via `inspect`), `call_agent` POSTs `/message:send` with `A2A-Version: 1.0` header + `ROLE_USER` enum + `{"task": ...}` unwrap; `AgentUnavailableError` on `httpx.HTTPError` and `TASK_STATE_FAILED` (unit-mocked + live); timeouts from Settings (global override → per-agent 120/60/30s defaults) |
| A2A-05 | 04-02 | SATISFIED | `graph.py` imports all 9 A2A-wrapped agents from `a2a_client` (AST check: no direct `agents.<name>` imports remain); `build_graph()`/`build_debate_graph()` signatures unchanged (verified); every A2A invocation goes through wrappers |
| A2A-05b | 04-02 | SATISFIED | `debate_node` partial-debate (D-05): `real_down`/`fake_down` flags, both-down → `exit_reason="agent_unavailable"`, one-sided skips convergence, IN-07 non-None appends — exercised with mocked advocates (both-down, one-sided ×2, converged) |

## Plan 04-01 must_haves

| # | Must-have | Status | Evidence |
| - | --------- | ------ | -------- |
| 1 | `a2a_client.py` with all 10 typed wrappers | ✓ | import + signature check |
| 2 | Each wrapper → SendMessageRequest → `POST /message:send` via `httpx.Client` → deserialized dict | ✓ | live call to search_agent returned diff (12 evidence, 3 queries); mocked COMPLETED/FAILED/WORKING paths |
| 3 | `AgentUnavailableError` on `httpx.HTTPError` / `TASK_STATE_FAILED` | ✓ | mocked paths raise with `[name:port] unavailable: cause` |
| 4 | `@degrade_on_unavailable` catches → per-agent degrade diff | ✓ | all 10 wrappers degrade-verified; shallow-copy isolation confirmed |
| 5 | `search_agent` rebuilds `EvidenceGraph` locally (D-03) | ✓ | rebuild yields 2-node graph from evidence list |
| 6 | `Settings` gains `a2a_client_timeout` + per-agent fields | ✓ | 11 fields; defaults 120/60/30; global None when unset |
| 7 | All 10 wrappers importable from `factcheck_agents.a2a_client` | ✓ | `from factcheck_agents.a2a_client import *` OK |

## Plan 04-02 must_haves

| # | Must-have | Status | Evidence |
| - | --------- | ------ | -------- |
| 1 | No direct imports of A2A-wrapped agents in `graph.py` | ✓ | AST scan — zero matches; `route_after_agreement` still from `agents.agreement_gate` |
| 2 | `build_graph()`/`build_debate_graph()` signatures/returns unchanged | ✓ | `(checkpointer=None)`; both return `CompiledStateGraph` |
| 3 | `debate_node` per-advocate `AgentUnavailableError` + partial-debate (D-05) | ✓ | behavioral asserts (see A2A-05b) |
| 4 | `AgentUnavailableError` → graceful degrade, pipeline continues | ✓ | CLI all-down smoke: exit 0, UNVERIFIED / 0.0 confidence; graceful-degrade unit test passes |
| 5 | Test patch targets updated | ⚠ DEVIATED | Plan's `a2a_client.*` targets are ineffective with `from-import` bindings; effective `graph.*` targets retained (documented deviation, REVIEW.md + SUMMARYs) — functional outcome verified |
| 6 | Existing tests pass unchanged | ✓ (parity) | 104 collected: 99 passed / 5 failed — EXACT baseline parity; the 5 failures are documented pre-existing (Phase 3 VERIFICATION.md corroborates), zero regressions, no ImportError |

## Behavior Assertions

- [x] `build_graph()` + `build_debate_graph()` compile and invoke (mocked A2A) → verdict dicts
- [x] `debate_node` both advocates down → `debate_exit_reason="agent_unavailable"`
- [x] `debate_node` one advocate down → partial debate (available side full turns, no convergence)
- [x] `debate_node` both available → convergence preserved (`exit_reason="converged"`)
- [x] CLI all 10 servers up → exit 0, valid verdict, **58s < 60s** (ROADMAP criterion 4a)
- [x] CLI all servers down → exit 0, UNVERIFIED / 0.0, no crash (ROADMAP criterion 4b)

## Gaps / Notes

- **None blocking.** Deviations (patch targets, protocol compliance fixes,
  environment-drift blockers) are documented in 04-01/04-02 SUMMARYs and
  04-REVIEW.md; all were necessary for the phase's goal and verified.
- Pre-existing failures (5) tracked in `deferred-items.md`; `run_fact_check()`
  and `mcp_server.py` still crash on the langgraph thread_id requirement
  (out of phase scope, logged).
- Security gate: `workflow.security_enforcement=true`, no `04-SECURITY.md` yet —
  run `/gsd-secure-phase 4` before advancing.
- Nyquist validation hook active (`workflow.nyquist_validation=true`) — run
  `/gsd-validate-phase 4` for the validation audit.

## Conclusion

**Phase 04 goal achieved:** LangGraph nodes invoke all 10 agents over A2A HTTP
with graceful degradation; end-to-end behavior proven live (servers up: real
verdict in 58s; servers down: UNVERIFIED degrade). Requirement IDs A2A-04,
A2A-05, A2A-05b are marked Complete in REQUIREMENTS.md and trace to passing
verification evidence.
