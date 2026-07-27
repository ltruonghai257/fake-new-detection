# Phase 6: LangGraph Wiring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-27
**Phase:** 06-LangGraph Wiring
**Areas discussed:** Checkpointer DB path, build_graph() signature, evaluate_agent cleanup scope, thread_id wiring boundary

---

## Checkpointer DB Path

### Q1: Where does the SqliteSaver .sqlite file path come from?

| Option | Description | Selected |
|--------|-------------|----------|
| Env var with default | FACTCHECK_CHECKPOINT_DB env var, default: .factcheck_checkpoints.db | ✓ |
| Hardcoded in graph.py | Hardcode the path directly in build_graph() | |
| Passed as argument | build_graph(db_path='.factcheck_checkpoints.db') | |

**User's choice:** Env var with default
**Notes:** Consistent with all existing config pattern.

---

### Q2: When SqliteSaver import fails, how should the fallback behave?

| Option | Description | Selected |
|--------|-------------|----------|
| Silent fallback to MemorySaver | except ImportError → MemorySaver, no noise | ✓ |
| Warning + fallback | logger.warning() then fall back | |

**User's choice:** Silent fallback to MemorySaver
**Notes:** Consistent with graceful-degrade pattern throughout the codebase.

---

## build_graph() Signature

### Q3: What is build_graph()'s new signature?

| Option | Description | Selected |
|--------|-------------|----------|
| Optional arg: build_graph(checkpointer=None) | Creates default internally; callers can pass MemorySaver() for tests | ✓ |
| Parameter-free (internal only) | Always creates its own checkpointer; tests must mock env | |

**User's choice:** build_graph(checkpointer=None)
**Notes:** No breaking change to existing call sites; clean test seam for Phase 8.

---

### Q4: Does initial_state() need to change in Phase 6?

| Option | Description | Selected |
|--------|-------------|----------|
| No change — thread_id is invoke config | thread_id lives in config dict, not FactCheckState | ✓ |
| Add thread_id to initial_state() | Phase 6 owns thread_id story end-to-end | |

**User's choice:** No change
**Notes:** Phase 7 handles wiring thread_id into CLI/API/MCP invoke calls.

---

## evaluate_agent Cleanup Scope

### Q5: What does Phase 6 do with evaluate_agent in agents/__init__.py?

| Option | Description | Selected |
|--------|-------------|----------|
| Leave __init__.py alone | Only remove from graph.py import; module stays exported | ✓ |
| Remove from __init__.py too | Complete cleanup in Phase 6 | |

**User's choice:** Leave __init__.py alone
**Notes:** Module is still valid code, just unwired.

---

### Q6: Should graph.py's module docstring be updated?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — update docstring | Change to "Search -> Verify -> (Social?) -> Conclusion" | ✓ |
| No — skip for now | Leave for later | |

**User's choice:** Yes
**Notes:** Small change, keeps docs accurate.

---

## thread_id Wiring Boundary

### Q7: Where does Phase 6's thread_id responsibility end?

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 6: graph only, Phase 7: callers | Phase 6 adds checkpointer; Phase 7 wires thread_id into callers | ✓ |
| Phase 6 does everything | Phase 6 also updates cli.py, __init__.py, mcp_server.py | |

**User's choice:** Phase 6: graph only
**Notes:** Clean phase boundary; avoids overlap with Phase 7 Output Surface scope.

---

### Q8: What should Phase 6 do about callers invoking without thread_id?

| Option | Description | Selected |
|--------|-------------|----------|
| Test + document: verify no crash | Confirm g.invoke(state) without config doesn't crash; add comment for Phase 7 | ✓ |
| Generate UUID fallback in build_graph | Wrap invoke with auto-UUID; changes call contract | |

**User's choice:** Test + document
**Notes:** Executor verifies behavior and documents finding as "# Note for Phase 7:" in graph.py.

---

## Claude's Discretion

None — all areas had a clear user decision.

## Deferred Ideas

None — discussion stayed within phase scope.
