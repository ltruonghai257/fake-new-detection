---
phase: "05"
status: clean
reviewed_at: 2026-08-22
depth: standard
findings_critical: 0
findings_warning: 2
findings_info: 0
findings_fixed: 2
---

# Code Review: Phase 05 — Demo App + Tests

## Summary

All 5 changed files reviewed. 2 warnings found and fixed inline (socket resource leaks in test helpers). No critical issues. Implementation matches plan specifications exactly.

## Files Reviewed

| File | Status | Findings |
|------|--------|----------|
| `demo_app/backend/streaming.py` | ✓ Clean | None |
| `tests/factcheck_agents/test_agent_http.py` | ✓ Fixed | 1 Warning (fixed) |
| `tests/factcheck_agents/test_a2a_integration.py` | ✓ Fixed | 1 Warning (fixed) |
| `factcheck_agents/__init__.py` | ✓ Clean | None |
| `factcheck_agents/mcp_server.py` | ✓ Clean | None |

## Findings

### W-01 — Socket resource leak in `_port_in_use()` (test_agent_http.py:37-41)
**Severity:** Warning (Fixed)
**File:** `tests/factcheck_agents/test_agent_http.py` lines 37–41

The original `_port_in_use()` created a raw `socket.socket()` without a context manager. If an exception occurred between creation and `s.close()`, the socket would leak.

**Fix applied:** Replaced with `with socket.socket() as s:` context manager pattern.

### W-02 — Socket resource leak in `a2a_agent_servers` fixture (test_a2a_integration.py:50-52)
**Severity:** Warning (Fixed)
**File:** `tests/factcheck_agents/test_a2a_integration.py` lines 50–52

Same issue: bare `socket.socket()` + manual `s.close()` without exception safety in the session fixture's port-conflict check.

**Fix applied:** Replaced with `with socket.socket() as s:` pattern, preserving the `result` variable for the subsequent `if result == 0` guard.

## Info Notes

- **streaming.py**: stage_error block correctly inserted after `accumulated.update(node_output)`, before `stage = NODE_STAGE_MAP.get(node_name)`. Vietnamese message matches plan exactly. `done.set()` called with no arguments.
- **test_agent_http.py**: All 10 tests use `daemon=True` threads, `server.should_exit = True` teardown, `A2A-Version: 1.0` header, and `TASK_STATE_COMPLETED` assertion.
- **test_a2a_integration.py**: Session fixture starts 8 agents with readiness polling. Both Vietnamese claims present. `build_debate_graph(checkpointer=None)` usage correct.
- **__init__.py / mcp_server.py**: `import uuid` at module level. `str(uuid.uuid4())` per-call. Signatures unchanged.

## Deviations from Plan

None. One plan typo noted: plan 05-02 task 05-02-01 referenced `PHOBERT_CHECKPOINT_PATH` env var, but the implementation correctly uses `VIFACTCHECK_CKPT_DIR` from `config.py` (plan had stale name from an earlier iteration).
