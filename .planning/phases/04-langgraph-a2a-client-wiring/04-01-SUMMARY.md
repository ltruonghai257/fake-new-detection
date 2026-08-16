---
phase: 04-langgraph-a2a-client-wiring
plan: '04-01'
subsystem: api
tags: [a2a, httpx, client, config, graceful-degrade]
requires:
  - phase: '03'
    provides: A2A agent servers on ports 9001-9010, serialize_state/deserialize_state helpers
provides:
  - a2a_client.py with AgentUnavailableError, sync call_agent, degrade_on_unavailable, 10 typed wrappers
  - per-agent A2A client timeout settings in config.py
affects: [04-02 graph.py wiring, MCP server, CLI]
actuals:
  tokens: 15000
  tasks: 5
  commits: 5
tech-stack:
  added: [httpx (already a project dep)]
  patterns:
    - "A2A client bridge: sync httpx.Client POST /message:send → deserialized state diff"
    - "degrade_on_unavailable decorator: catch AgentUnavailableError → per-agent degrade diff"
    - "Per-agent timeout resolution: global override → per-agent Settings field"
key-files:
  created: [factcheck_agents/a2a_client.py]
  modified: [factcheck_agents/config.py]
key-decisions:
  - "D-01 sync bridge via httpx.Client (no asyncio shim)"
  - "D-03 EvidenceGraph rebuilt locally in search_agent wrapper from returned evidence list"
  - "TASK_STATE_WORKING response treated as empty diff with warning (Risk 2 mitigation)"
  - "Pre-existing test failures logged to deferred-items.md, not fixed (scope boundary)"
patterns-established:
  - "Wrapper pattern: @degrade_on_unavailable('<name>', _DEGRADE_<NAME>) + one-liner call_agent"
requirements-completed: [A2A-04]
coverage:
  - id: D1
    description: a2a_client.py with AgentUnavailableError, call_agent (httpx sync POST /message:send), _extract_artifact_data, _timeout_for
    requirement: A2A-04
    verification:
      - kind: unit
        ref: "command: python -c imports + mocked httpx error/FAILED/WORKING/COMPLETED paths"
        status: pass
    human_judgment: false
  - id: D2
    description: degrade_on_unavailable decorator + 10 per-agent degrade diffs
    requirement: A2A-04
    verification:
      - kind: unit
        ref: "command: decorator returns degrade diff on AgentUnavailableError, passes through success, shallow-copy isolation"
        status: pass
    human_judgment: false
  - id: D3
    description: 10 typed wrapper functions, search_agent rebuilds EvidenceGraph from evidence (D-03)
    requirement: A2A-04
    verification:
      - kind: unit
        ref: "command: all wrappers importable; search_agent rebuild yields 2-node graph; per-agent degrade checks"
        status: pass
    human_judgment: false
  - id: D4
    description: config.py Settings gains a2a_client_timeout (optional) + 10 per-agent timeout fields (120/60/30 defaults)
    requirement: A2A-04
    verification:
      - kind: unit
        ref: "command: settings.a2a_client_timeout_search == 120, evaluate == 60, agreement_gate == 30"
        status: pass
    human_judgment: false
  - id: D5
    description: No new test-suite failures from a2a_client additions (5 pre-existing failures confirmed at baseline)
    requirement: A2A-04
    verification:
      - kind: unit
        ref: "command: python -m pytest tests/factcheck_agents/ (99 passed, 5 failed — all 5 reproduce at HEAD~5)"
        status: pass
    human_judgment: false
duration: 14min
completed: 2026-08-16
status: complete
---

# Phase 04 Plan 01: Implement `a2a_client.py` Summary

**Sync A2A HTTP client with 10 degrade-decorated typed wrappers and per-agent timeout settings, ready for graph.py wiring in 04-02**

## Performance

- **Duration:** 14 min
- **Started:** 2026-08-16T18:13:33Z
- **Completed:** 2026-08-16T18:27:32Z
- **Tasks:** 5
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- `factcheck_agents/a2a_client.py` — `AgentUnavailableError`, sync `call_agent()` via `httpx.Client` posting `POST /message:send`, `_extract_artifact_data`, `_timeout_for`
- `@degrade_on_unavailable` decorator factory with all 10 per-agent `_DEGRADE_*` diffs (shallow-copied on return)
- 10 typed wrapper functions (search, evaluate, real_source, fake_source, social_loop, agreement_gate, real/fake_advocate, judge, conclusion); `search_agent` rebuilds `EvidenceGraph` locally (D-03)
- `config.py` — `a2a_client_timeout` global override + 10 per-agent timeout fields (LLM 120s, evaluate 60s, social_loop/agreement_gate 30s)
- Verified all error paths: HTTP error → `AgentUnavailableError`, `TASK_STATE_FAILED` → `AgentUnavailableError`, `TASK_STATE_WORKING` → empty diff + warning, COMPLETED → artifact data

## Task Commits

Each task was committed atomically:

1. **Task 1: A2A client timeout settings** - `fb4e306` (feat)
2. **Task 2: a2a_client with call_agent + AgentUnavailableError** - `c423e8b` (feat)
3. **Task 3: degrade_on_unavailable + 10 degrade diffs** - `8fe7ace` (feat)
4. **Task 4: 10 typed wrapper functions** - `e63a713` (feat)
5. **Task 5: verification + deferred items log** - `70b334e` (docs)

**Plan metadata:** (pending commit)

## Files Created/Modified

- `factcheck_agents/a2a_client.py` - A2A client: error class, sync call_agent, timeout resolution, degrade decorator, 10 wrappers
- `factcheck_agents/config.py` - 11 new `a2a_client_timeout*` Settings fields
- `.planning/phases/04-langgraph-a2a-client-wiring/deferred-items.md` - pre-existing test failures logged

## Decisions Made

- Used `httpx.Client` sync bridge (D-01) — no asyncio shim, zero caller changes
- Timeout resolution: global `A2A_CLIENT_TIMEOUT` (optional) wins; otherwise per-agent Settings field with hardcoded default (no module fallback dict — Settings is single source of truth)
- `TASK_STATE_WORKING` response → logged warning + empty diff (simple Risk-2 mitigation; no polling)
- Shallow copy `dict(degrade_diff)` on degrade return to protect module constants from mutation

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

**Plan-command note:** The 6 verification commands in Task 04-01-05 used one-liner `python -c` forms with decorators/`with` blocks, which are invalid Python one-line syntax. Re-run in multi-line form; assertions identical. Noted for future plan generation.

---

**Total deviations:** 0 auto-fixed
**Impact on plan:** None

## Issues Encountered

- 5 pre-existing test failures confirmed at baseline (HEAD~5, before this plan's commits): `test_agreement_unavailable_model_treated_as_zero`, `test_map_to_binary_unverified_maps_to_fake`, `test_llm_unverified_maps_to_fake`, `test_worldcup_claim`, `test_nei_short_circuit`. All unrelated to A2A client (label-mapping expectation drift). Logged to `deferred-items.md`; the suite collects 104 tests (plan's "83" count is stale) — 99 pass.
- `.venv` is used for Python (system `python` not on PATH).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ready for 04-02: graph.py import refactor + debate_node partial-debate semantics + test patch target updates
- `from factcheck_agents.a2a_client import <name>` works for all 10 wrappers

---

*Phase: 04-langgraph-a2a-client-wiring*
*Completed: 2026-08-16*
