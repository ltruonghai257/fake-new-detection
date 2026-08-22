---
plan: "05-02"
phase: 5
status: complete
completed_at: 2026-08-22T10:15:00Z
duration_minutes: 16
tasks_completed: 5
tasks_total: 5
deviations: []
key-files:
  created:
    - tests/factcheck_agents/test_agent_http.py
    - tests/factcheck_agents/test_a2a_integration.py
  modified:
    - factcheck_agents/__init__.py
    - factcheck_agents/mcp_server.py
    - .planning/phases/05-demo-app-tests/05-VALIDATION.md
---

# Plan 05-02: Tests + Backward Compatibility Fixes — Summary

## What Was Built

1. **`tests/factcheck_agents/test_agent_http.py`** — 10 `@pytest.mark.integration` HTTP tests, one per TaskHandler. Each starts a real uvicorn server in a background thread, sends an A2A Task, asserts TASK_STATE_COMPLETED and agent-specific diff keys.

2. **`tests/factcheck_agents/test_a2a_integration.py`** — Session-scoped fixture starting 8 graph-path agents. 2 integration tests with Vietnamese claims (Hanoi capital, moon cheese).

3. **`factcheck_agents/__init__.py`** — Fixed thread_id crash in `run_fact_check()`. Added `import uuid` and `config={"configurable": {"thread_id": str(uuid.uuid4())}}` to `graph.invoke()`.

4. **`factcheck_agents/mcp_server.py`** — Fixed thread_id crash in `fact_check()`. Added `import uuid`, `thread_id = str(uuid.uuid4())` local var, passed `config={"configurable": {"thread_id": thread_id}}` to `graph.invoke()`.

5. **`05-VALIDATION.md`** — Updated to `wave_0_complete: true`, `nyquist_compliant: true`, `status: complete`.

## Deviations
None.

## Self-Check: PASSED

All must_haves verified:
1. ✓ `test_agent_http.py` exists with exactly 10 `@pytest.mark.integration` test functions
2. ✓ `test_a2a_integration.py` exists with `a2a_agent_servers` session fixture and 2 integration tests
3. ✓ `factcheck_agents/__init__.py` has `import uuid` and `config={"configurable": {"thread_id": str(uuid.uuid4())}}` in `graph.invoke()` call
4. ✓ `factcheck_agents/mcp_server.py` has `import uuid` and `config={"configurable": {"thread_id": thread_id}}` in `graph.invoke()` call
5. ✓ `pytest tests/ -m "not integration" -q` exits 0, 159 passed (≥ 99)
6. ✓ `05-VALIDATION.md` has `wave_0_complete: true`
