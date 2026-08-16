---
phase: 04-langgraph-a2a-client-wiring
plan: '04-02'
subsystem: api
tags: [a2a, langgraph, graph, debate, graceful-degrade, smoke-test]
requires:
  - phase: '04'
    provides: a2a_client.py wrappers (04-01)
  - phase: '03'
    provides: agent servers on ports 9001-9010, start/stop scripts
provides:
  - graph.py fully wired through a2a_client.* with partial-debate semantics
  - working end-to-end A2A pipeline (verified via live CLI smoke test)
affects: [Phase 5 demo app + tests, MCP server, run_fact_check]
actuals:
  tokens: 21000
  tasks: 7
  commits: 10
tech-stack:
  added: []
  patterns:
    - "A2A protocol compliance: A2A-Version header + ROLE_USER enum + {'task': ...} unwrap"
    - "Server-side EvidenceGraph rebuild from repr string (mirrors client D-03)"
    - "EvidenceGraph checkpoint persistence via _asdict()/graph_data constructor round-trip"
key-files:
  created: []
  modified:
    - factcheck_agents/graph.py
    - factcheck_agents/a2a_client.py
    - factcheck_agents/graph_utils.py
    - factcheck_agents/cli.py
    - factcheck_agents/agents/conclusion_agent.py
key-decisions:
  - "Patch targets stay on factcheck_agents.graph (a2a_client patches are ineffective with from-import bindings) — deviation from plan"
  - "EvidenceGraph made checkpointer-serializable (graph_data round-trip) — required for any real run"
  - "cli.py passes thread_id config — current langgraph requires it"
  - "No new test failures: suite at exact baseline parity (99 pass / 5 pre-existing fail)"
requirements-completed: [A2A-05, A2A-05b]
coverage:
  - id: D1
    description: graph.py routes every A2A-wrapped agent through a2a_client.* (9 wrappers + AgentUnavailableError); local nodes unchanged
    requirement: A2A-05
    verification:
      - kind: unit
        ref: "command: AST check — no direct agent imports; build_graph/build_debate_graph compile"
        status: pass
    human_judgment: false
  - id: D2
    description: debate_node partial-debate semantics (D-05): per-side down flags, both-down exit_reason agent_unavailable, one-sided skips convergence, IN-07 non-None appends
    requirement: A2A-05b
    verification:
      - kind: unit
        ref: "command: mocked both-down / one-sided / converged scenarios (behavioral asserts)"
        status: pass
    human_judgment: false
  - id: D3
    description: Existing test suite at baseline parity (99 passed / 5 pre-existing failures, zero regressions; no ImportError)
    requirement: A2A-05
    verification:
      - kind: unit
        ref: "command: python -m pytest tests/factcheck_agents/ — 99 passed, 5 failed (identical to baseline)"
        status: pass
    human_judgment: false
  - id: D4
    description: CLI smoke test — all 10 A2A servers up: exit 0, valid verdict, 58s < 60s
    requirement: A2A-05
    verification:
      - kind: e2e
        ref: "command: python -m factcheck_agents.cli 'Tin tức test' with scripts/start_agents.sh running"
        status: pass
    human_judgment: false
  - id: D5
    description: CLI smoke test — all servers down: exit 0, UNVERIFIED verdict, confidence 0.0, no crash
    requirement: A2A-05
    verification:
      - kind: e2e
        ref: "command: python -m factcheck_agents.cli 'Tin tức test' with agents stopped"
        status: pass
    human_judgment: false
duration: 74min
completed: 2026-08-17
status: complete
---

# Phase 04 Plan 02: Refactor `graph.py` + Test Updates Summary

**LangGraph pipeline fully wired through A2A HTTP with partial-debate semantics, proven end-to-end by live CLI smoke tests (58s verdict with servers up, graceful UNVERIFIED with servers down)**

## Performance

- **Duration:** 74 min
- **Started:** 2026-08-16T19:02:00Z (approx.)
- **Completed:** 2026-08-17T00:16:00Z (approx.)
- **Tasks:** 7
- **Files modified:** 5 source files + 1 test file (reverted) + planning docs

## Accomplishments

- `graph.py` — all 9 A2A-wrapped agents imported from `a2a_client`; `social_search_agent`, `verify_agent`, `expert_agent`, `route_after_agreement`, `reranker_node` stay local; `build_graph()`/`build_debate_graph()` signatures unchanged
- `debate_node` — partial-debate semantics (D-05): `real_down`/`fake_down` flags, both-down → `exit_reason="agent_unavailable"`, one-sided debates skip convergence, IN-07 non-None turn appends
- **Live CLI smoke test (Task 04-02-07) passed both directions**, requiring 3 real fixes found by the smoke test:
  - A2A protocol compliance in `call_agent`: `A2A-Version: 1.0` header, `ROLE_USER` enum, `{"task": ...}` response unwrap (SDK 1.1.2 protobuf REST transport)
  - Server-side `conclusion_agent` rebuilds `EvidenceGraph` from evidence when it arrives as a repr string over A2A
  - Two environment-drift blockers: cli.py `thread_id` config + `EvidenceGraph` checkpoint serialization (`_asdict()`/`graph_data` round-trip)
- Test suite at **exact baseline parity**: 99 passed / 5 pre-existing failures (104 collected; plan's "83" count stale), zero regressions

## Task Commits

1. **Task 1: graph.py import refactor** - `b06c406` (refactor)
2. **Task 2: debate_node partial-debate** - `a2b53f3` (feat)
3. **Task 3: test_graceful_degrade patch targets** - `5b660aa` (test) → reverted `536eb47` (fix, see deviations)
4. **Task 4: test_debate_pipeline_integration split** - (reverted; deviation, see below)
5. **Task 5: full suite verification** - `536eb47` + user-applied identical fix (no net change)
6. **Task 6: smoke imports + AST check** - no commit needed (verification only)
7. **Task 7: CLI smoke test** - `7addca6` (thread_id + EvidenceGraph serde), `919c5f0` (A2A protocol + conclusion rebuild), `7a9dcad` (deferred items)

**Plan metadata:** (pending commit)

## Files Created/Modified

- `factcheck_agents/graph.py` - A2A wrapper imports, partial-debate debate_node, module logger
- `factcheck_agents/a2a_client.py` - A2A-Version header, ROLE_USER, task unwrap
- `factcheck_agents/graph_utils.py` - EvidenceGraph `_asdict()`/`graph_data` checkpoint round-trip
- `factcheck_agents/cli.py` - `thread_id` config on invoke
- `factcheck_agents/agents/conclusion_agent.py` - EvidenceGraph rebuild from repr string
- `.planning/phases/04-langgraph-a2a-client-wiring/deferred-items.md` - expanded records

## Decisions Made

- **Patch targets stay on `factcheck_agents.graph`** for A2A-wrapped agents. RESEARCH Pattern 5 assumed patching `a2a_client.*` intercepts graph node calls; with `from .a2a_client import X` bindings it does not (real wrappers hit localhost and degrade). `graph.*` patches intercept at the node call site. Test files reverted to the effective original targets.
- **EvidenceGraph checkpoint serialization** via jsonplus constructor-kwargs protocol (`_asdict()` returning plain nodes/edges; `__init__(graph_data=...)` rebuilds). Needed for ANY real run (social_search always injects the graph; SqliteSaver/MemorySaver both msgpack-serialize state).
- **cli.py** passes `config={"configurable": {"thread_id": ...}}` — current langgraph requires it; external CLI interface unchanged.
- Smoke-test-driven protocol fixes adopted as the canonical client contract (header + enum + unwrap).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] a2a_client patch targets are ineffective — reverted to graph namespace**
- **Found during:** Task 4 (test_debate_pipeline_integration split)
- **Issue:** Plan (RESEARCH Pattern 5) mandated splitting patches to `factcheck_agents.a2a_client` for real_source/fake_source/judge. With `from .a2a_client import X` bindings in graph.py, patching the a2a_client module does not replace graph's bindings — the real wrappers run and degrade (or hit live servers). test_graceful_degrade passed only by accident (degrade path still yields a verdict dict).
- **Fix:** Reverted both test files to single-block `patch.multiple("factcheck_agents.graph", ...)` (and `@patch("factcheck_agents.graph.*")`), which intercepts at node call sites. User independently applied the identical fix to test_debate_pipeline_integration.py.
- **Files modified:** tests/factcheck_agents/test_graceful_degrade.py (reverted), test_debate_pipeline_integration.py (no net change)
- **Verification:** full suite at baseline parity
- **Committed in:** `536eb47`

**2. [Rule 3 - Blocking] CLI cannot run: langgraph requires thread_id**
- **Found during:** Task 7 (CLI smoke test)
- **Issue:** Pre-existing (confirmed at HEAD~8): `graph.invoke()` without `thread_id` raises ValueError. Blocked the entire smoke test.
- **Fix:** cli.py passes `config={"configurable": {"thread_id": f"cli-{hash(...)}"}}`. run_fact_check()/mcp_server.py still crash (logged to deferred-items).
- **Files modified:** factcheck_agents/cli.py
- **Committed in:** `7addca6`

**3. [Rule 3 - Blocking] Checkpointer cannot serialize EvidenceGraph**
- **Found during:** Task 7 (CLI smoke test)
- **Issue:** Pre-existing: social_search_agent injects an `EvidenceGraph` instance; the langgraph checkpointer's msgpack serde raises TypeError. MemorySaver crashes identically. Tests never hit it (agents mocked).
- **Fix:** `EvidenceGraph._asdict()` + `__init__(graph_data=...)` round-trip via jsonplus constructor-kwargs protocol.
- **Files modified:** factcheck_agents/graph_utils.py
- **Committed in:** `7addca6`

**4. [Rule 1 - Bug] A2A protocol compliance (3 sub-fixes)**
- **Found during:** Task 7 (CLI smoke test, servers up)
- **Issue:** SDK 1.1.2 REST transport rejected the client: missing `A2A-Version` header (defaults to v0.3 → 400), role must be `ROLE_USER` enum (not "user"), and responses wrap the task in `{"task": ...}` so diffs came back empty.
- **Fix:** Header + enum + unwrap in `call_agent`; server-side `conclusion_agent` rebuilds EvidenceGraph from evidence when it arrives as a repr string (over A2A the graph cannot travel as an object).
- **Files modified:** factcheck_agents/a2a_client.py, factcheck_agents/agents/conclusion_agent.py
- **Committed in:** `919c5f0`

**5. [Plan defect - verification syntax] 04-01-05 one-liner commands invalid** — `@decorator; def` and `with` blocks cannot appear in `;`-chained `python -c`; re-run multi-line with identical assertions.

---

**Total deviations:** 4 auto-fixed (2 blocking-environment, 2 bug-fix) + 1 plan-defect note
**Impact on plan:** All fixes required for the phase's core deliverable (end-to-end A2A pipeline) to function at all. No scope creep; external interfaces unchanged.

## Issues Encountered

- The 5 pre-existing test failures are unchanged (baseline parity). See deferred-items.md.
- `run_fact_check()` and `mcp_server.py` still crash on the thread_id requirement — logged to deferred-items.md, out of 04-02 scope.
- langgraph warns EvidenceGraph is an unregistered msgpack type (future strict mode) — noted in deferred-items.md.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 5 (Demo App + Tests) can build on a working end-to-end A2A pipeline; `streaming.py`/MCP/`run_fact_check` need the thread_id fix (or switch to checkpointer-less graphs) — flagged.
- ROADMAP success criterion 4 (CLI up/down smoke tests) verified live.

---

*Phase: 04-langgraph-a2a-client-wiring*
*Completed: 2026-08-17*
