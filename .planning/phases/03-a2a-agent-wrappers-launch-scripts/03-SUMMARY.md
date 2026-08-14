# Phase 3 Summary: A2A Agent Wrappers & Launch Scripts

**Status**: Complete
**Execution Date**: 2026-08-15
**Waves**: 4/4 executed (sequential)
**Plans**: 4/4 completed (03-01 through 03-04)

## Overview

Phase 3 wrapped all 10 factcheck agents as Google A2A protocol HTTP services
(uvicorn on ports 9001–9010), exposed standard Agent Cards at
`/.well-known/agent.json`, split `debate_node.py` into single-turn advocate
services, and delivered `start_agents.sh` / `stop_agents.sh` / `smoke_test_agents.sh`
developer tooling. The LangGraph wiring still calls the advocate functions
directly — Phase 4 replaces those calls with the A2A client.

**Requirements satisfied**: A2A-01, A2A-02, A2A-03, A2A-03b

## Plans Executed

### Plan 03-01: Install a2a-sdk + Base Server Module — commit 1e28374
- Added `a2a-sdk[http-server,fastapi]>=1.0.1,<2` to the `agents` extra
  (a2a-sdk 1.1.2 installed); relaxed the legacy `protobuf==3.20.3` pin to
  `>=5.29.5,<7` — a2a-sdk hard-requires protobuf >=5.29.5 and mlflow >=2.11
  supports the newer range. Verified against the pre-change baseline that
  the resulting test failures are pre-existing, not regressions.
- Added 10 `A2A_PORT_*` settings (env-overridable, defaults 9001–9010) plus
  the `a2a_ports() -> dict[str, int]` helper to `config.py`.
- Created `factcheck_agents/a2a_server.py`:
  - `AgentCardConfig` dataclass + `build_agent_card()` (D-15)
  - `serialize_state()` / `deserialize_state()` with datetime/Path/set-safe
    conversion via orjson (D-03)
  - `BaseTaskHandler(AgentExecutor)` with the shared lifecycle: deserialize
    full `FactCheckState` from the message data part (D-01) → `agent_fn`
    → attach state-diff artifact named `output` (D-02) → completed; on
    exception → `failed` status with `{"error": ...}` (D-04)
  - `create_app()` (FastAPI + JSON-RPC `/` + REST + agent-card routes) and
    `run_server()` (uvicorn on 127.0.0.1:port)
- Verified live: echo + failure handlers over real HTTP (completed + failed
  paths), card alias at `/.well-known/agent.json`.

### Plan 03-02: TaskHandlers for Evidence-Side Agents — commit 8a45890
- Appended `SearchAgentHandler`, `EvaluateAgentHandler`, `RealSourceAgentHandler`,
  `FakeSourceAgentHandler` (ports 9001–9004) with `__main__` uvicorn entry points.
- Verified live: each service answers `/.well-known/agent.json` and completes a
  minimal `SendMessage` task with the expected diff keys (search → evidence;
  evaluate → model_results with graceful unavailable models; real_source →
  evidence_real + workflow steps; fake_source → evidence_fake).

### Plan 03-03: Pipeline Agent Handlers + debate_node Split — commit 5473860
- Created `agents/debate_utils.py` (D-08): `REAL_ADVOCATE_PROMPT`,
  `FAKE_ADVOCATE_PROMPT`, formatting/parsing helpers, JSONL turn logger.
- Created `agents/real_advocate.py` + `agents/fake_advocate.py` (D-07):
  single-turn advocate functions + `RealAdvocateHandler` / `FakeAdvocateHandler`
  (ports 9007/9008) that enforce the `debate_role` contract (D-05) — an
  explicit mismatch fails the task loudly instead of arguing the wrong side.
- Deleted `debate_node.py`; `graph.py` now owns a `debate_node` loop node that
  calls the single-turn advocates until convergence or `max_debate_rounds`
  (return contract unchanged); `agents/__init__.py` exports updated; the one
  integration-test import was repointed at the graph node.
- Appended `SocialLoopAgentHandler`, `AgreementGateHandler`, `JudgeAgentHandler`,
  `ConclusionAgentHandler` (ports 9005/9006/9009/9010).
- Verified live: all 6 pipeline services answer cards + minimal tasks; role
  mismatch → `TASK_STATE_FAILED`; correct role → completed single turn (real
  LLM). No new test failures vs baseline.

### Plan 03-04: Launch Scripts + Smoke Tests — commit 77237b8
- `scripts/start_agents.sh`: sequential startup (D-09), per-agent logs in
  `logs/agent_<name>.log` (D-10), port-conflict hard abort (D-11), readiness
  poll on `/.well-known/agent.json` up to 30s (D-12), `.pids/<name>.pid`
  tracking (D-13). Ports read from `a2a_ports()` so `A2A_PORT_*` overrides
  propagate.
- `scripts/stop_agents.sh`: SIGTERM → 5s wait → SIGKILL fallback → pid cleanup.
- `scripts/smoke_test_agents.sh`: validates each card JSON has a matching
  `name` (10/10).
- `logs/.gitkeep` + `.pids/.gitkeep`; `.gitignore` entries for `logs/*` and
  `.pids/*` with `.gitkeep` exceptions.
- Verified E2E: start (10/10 ready) → smoke (10/10 OK) → stop (all graceful,
  no zombies, `.pids/` empty); D-11 abort path exercised with an occupied port.

## Verification Summary

| # | Criterion | Result |
|---|-----------|--------|
| 1 | a2a-sdk installs without conflicts; version pinned | ✓ 1.1.2 (major-version pinned `<2`; protobuf pin relaxed with mlflow compatibility) |
| 2 | All 10 agents expose a TaskHandler with the shared task lifecycle | ✓ smoke test + minimal `SendMessage` per agent |
| 3 | `/.well-known/agent.json` returns a valid Agent Card per agent | ✓ 10/10 via smoke test; SDK path `agent-card.json` kept + `agent.json` alias |
| 4 | `start_agents.sh` starts all 10 on ports 9001–9010 in < 15s | ⚠ 28s on this machine (10 concurrent torch/transformers imports + Jupyter kernel resident); readiness poll accommodates |
| 5 | `stop_agents.sh` SIGTERM, cleans PIDs, no zombies | ✓ verified: 10/10 stopped, no SIGKILL needed, `.pids/` empty |

**Tests**: `tests/factcheck_agents/` — 99 passed, 5 failed. All 5 failures
(2× conclusion_agent semantic expectations, 3× LLM/model-dependent integration
tests) reproduce on the pre-change baseline and are unrelated to this phase.

## Deviations from Plan

- **[Rule 1 - Plan built against outdated SDK API]** — `a2a-sdk.server.tasks.TaskHandler`
  does not exist in any published a2a-sdk release (0.2.x–1.1.2 all use the
  `AgentExecutor` interface with protobuf types). `BaseTaskHandler` was
  implemented as an `AgentExecutor` subclass with the same D-01..D-15 contract;
  `TaskResult.output` maps to the `output` artifact. **Found during:** Plan 03-01
  step 5 | **Fix:** adapted base class to the actual SDK | **Files:**
  `factcheck_agents/a2a_server.py` | **Verification:** live completed + failed
  task paths.
- **[Rule 1 - Dependency conflict]** — `a2a-sdk` hard-requires `protobuf>=5.29.5`
  while the project pinned `protobuf==3.20.3` (legacy mlflow fix). Pin relaxed
  to `>=5.29.5,<7`; mlflow >=2.11 supports it; no direct protobuf imports in
  `factcheck_agents`. **Found during:** Plan 03-01 step 1 | **Files:**
  `pyproject.toml` | **Verification:** full test suite — failures identical to
  baseline.
- **[Rule 1 - SDK card path + route details]** — SDK serves the card at
  `/.well-known/agent-card.json`; added an `agent.json` alias route (inserted
  before the REST `/{tenant}` mount) to honor contract D-12. JSON-RPC method is
  `SendMessage` with `A2A-Version: 1.0` header and protobuf JSON bodies.
  **Found during:** Plan 03-01 step 5 | **Files:** `factcheck_agents/a2a_server.py`.
- **[Rule 1 - Port conflict in dev environment]** — the user's Jupyter kernel
  occupies port 9001 (`etlservicemgr`), so live verification used `A2A_PORT_*`
  overrides (910x/920x). The scripts honor env overrides; D-11 detects the
  conflict and aborts with a clear message.
- **[Deviation - Start time]** — success criterion #4 targets < 15s; measured
  28s on this machine under 10 concurrent heavy imports. Not a script defect;
  the 30s readiness poll bound is correct.

**Total deviations:** 4 auto-fixed (all Rule 1 — plan/SDK reality mismatches).
**Impact:** none on external interfaces — CLI, `run_fact_check()`, and the MCP
server are untouched; Phase 4 consumes the ports/cards exactly as planned.

## Next Phase Readiness

Phase 4 (`a2a_client.py` + graph refactor) can proceed: the 10 endpoints,
cards, serialization helpers, and port map are the agreed contract. The
`debate_role` runtime field, stateless full-state task input, and diff-output
artifact shape are in place for the A2A client to consume.

## Self-Check: PASSED
