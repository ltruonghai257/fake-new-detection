---
phase: "05-demo-app-tests"
status: passed
verified_at: "2026-09-07T19:03:23Z"
must_haves_score: 9/9
---

# Phase 05 Verification Report

## Must-Haves Check

| # | Must-Have | Status | Evidence |
|---|-----------|--------|---------|
| 1 | `demo_app/backend/streaming.py` contains `"stage_error"` as event type in `_post()` call inside `run_graph()` | ✓ | `grep -n "stage_error" demo_app/backend/streaming.py` → line 167: `"type": "stage_error"` |
| 2 | `demo_app/backend/streaming.py` contains exact Vietnamese message string | ✓ | `grep -n "Một số dịch vụ" demo_app/backend/streaming.py` → line 169 confirms message |
| 3 | `stage_error` block appears AFTER `accumulated.update(node_output)` (line 157) and BEFORE `stage = NODE_STAGE_MAP.get(node_name)` (line 177) | ✓ | Lines confirmed: update=157, stage_error=163–174, NODE_STAGE_MAP=177 |
| 4 | `done.set()` (no arguments) called in stage_error path before `break` | ✓ | `grep -n "done.set()" streaming.py` → line 173 (no arguments); `done.set(True)` absent |
| 5 | Existing `except Exception as exc` block unchanged (`_post({"type": "error", "error": str(exc)})`) | ✓ | `grep -n '"type": "error"' streaming.py` → line 331 confirms unchanged |
| 6 | `tests/factcheck_agents/test_agent_http.py` exists with exactly 10 `@pytest.mark.integration` test functions | ✓ | `grep -c "@pytest.mark.integration"` → 10; `grep -c "^def test_"` → 10 |
| 7 | `tests/factcheck_agents/test_a2a_integration.py` exists with `a2a_agent_servers` session fixture and exactly 2 integration tests | ✓ | `grep -c "@pytest.mark.integration"` → 2; fixture at line 35; `scope="session"` at line 34 |
| 8 | `factcheck_agents/__init__.py` has `import uuid` (module level) and `config={"configurable": {"thread_id": str(uuid.uuid4())}}` in `graph.invoke()` | ✓ | `grep -n "import uuid"` → line 21; `grep -n "uuid.uuid4()"` → line 39 with config kwarg |
| 9 | `factcheck_agents/mcp_server.py` has `import uuid` (module level), `thread_id = str(uuid.uuid4())` local var, and `config={"configurable": {"thread_id": thread_id}}` in `graph.invoke()` | ✓ | `grep -n "import uuid"` → line 27; `grep -n "uuid.uuid4()"` → line 46; `thread_id` config at line 49 |

## Requirement Traceability

| Req ID | Plan | Status | Evidence |
|--------|------|--------|---------|
| A2A-06 | 05-01, 05-02 | SATISFIED | SSE bridge uses A2A client path; backward-compat regression gate passes (159 tests, per 05-01-SUMMARY and 05-02-SUMMARY); `_post({"type": "error", ...})` unchanged at line 331 |
| A2A-06b | 05-01 | SATISFIED | `stage_error` detection block present in `streaming.py` lines 158–174; scans `node_output.get("messages", [])` for `"unavailable"`; emits Vietnamese message; calls `done.set()` + `break`; HTTP closes 200 |
| A2A-07 | 05-02 | SATISFIED | `tests/factcheck_agents/test_agent_http.py` — 10 `@pytest.mark.integration` functions (one per TaskHandler); starts real uvicorn in daemon thread; sends real A2A Task via httpx with `"A2A-Version": "1.0"` header; asserts `TASK_STATE_COMPLETED`; port-conflict skip guard present |
| A2A-07b | 05-02 | SATISFIED | `tests/factcheck_agents/test_a2a_integration.py` — `a2a_agent_servers` session-scoped fixture starts 8 graph-path agents (ports 9003–9010); polls `/.well-known/agent.json` readiness; 2 integration tests with Vietnamese claims (`"Hà Nội là thủ đô của Việt Nam"` and `"Mặt trăng làm từ phô mai"`); teardown via `server.should_exit = True` |
| A2A-08 | 05-02 | SATISFIED | `factcheck_agents/__init__.py` line 38–39: `graph.invoke(state, config={"configurable": {"thread_id": str(uuid.uuid4())}})` — thread_id crash fixed; `factcheck_agents/mcp_server.py` lines 46–49: `thread_id = str(uuid.uuid4())` then `config={"configurable": {"thread_id": thread_id}}` — thread_id crash fixed; external function signatures unchanged |

## Gaps

None — all must-haves verified on disk.

## Notes

- REQUIREMENTS.md still shows A2A-06 through A2A-08 as `Pending` in the traceability table (the table was not updated by Phase 05). This is a documentation artefact only; the code and tests are implemented as required.
- VALIDATION.md task status column shows all tasks as `⬜ pending` (table was not updated post-execution), but frontmatter confirms `wave_0_complete: true`, `nyquist_compliant: true`, `status: complete` — authoritative state is the frontmatter.
- Manual-only verifications (SSE stage_error in live browser with agents stopped/started) are listed in VALIDATION.md and require human execution; they are out of scope for automated verification.
- 159 non-integration tests pass (confirmed by Phase 05 execution context and 05-01/05-02 SUMMARY files).
