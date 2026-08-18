# Phase 5: Demo App + Tests - Context

**Gathered:** 2026-08-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `stage_error` SSE event handling to `demo_app/backend/streaming.py` (A2A-06b — the graph is already wired to `a2a_client.*` from Phase 4, so A2A-06 is substantially complete); write 10 in-process agent HTTP tests (A2A-07, `@pytest.mark.integration`); add a new `test_a2a_integration.py` with session-scoped uvicorn fixtures for graph end-to-end tests (A2A-07b); fix `run_fact_check()` and `mcp_server.py` thread_id crash (A2A-08); confirm backward compat for all 3 entry points.

**Not in scope:** React frontend changes, changing SSE event types (byte-for-byte identical to v3.0 output), new LLM models, modifying `training/`, changing `a2a_client.py` or `a2a_server.py` architecture.

</domain>

<decisions>
## Implementation Decisions

### SSE Bridge — stage_error Handling (A2A-06b)

- **D-01:** Detect agent unavailability by scanning `node_output.get("messages", [])` for entries containing the string `"unavailable"`. The `@degrade_on_unavailable` decorator in `a2a_client.py` already injects these messages (e.g., `[Search] agent unavailable — degraded`) into the degrade diff; no new state fields or decorator changes needed. — **Reversibility:** reversible — local change inside `run_graph()` loop only.

- **D-02:** When unavailability is detected in a node's messages, streaming.py emits `{"type": "stage_error", "data": {"message": "<Vietnamese string>"}}` via `_post()`, then sets `done.is_set()` to close the stream immediately. The generator exits via the existing `break` path; HTTP response closes with 200 as required by A2A-06b. — **Reversibility:** reversible — inside `run_graph()` exception path only.

### Claude's Discretion — SSE stage_error message

- Vietnamese message content: generic fixed string `"Một số dịch vụ tạm thời không khả dụng. Không thể hoàn thành phân tích."` — no per-agent lookup map (simpler, satisfies A2A-06b without adding an agent-name translation table).

### Agent HTTP Tests (A2A-07)

- **D-03:** Each of the 10 agent `TaskHandler` tests starts a real `uvicorn.Server` instance in a background thread, sends a real A2A `Task` JSON payload via `httpx.Client` to `localhost:{port}`, and asserts `TaskResult` fields match expected types. No mocking of the agent function — true end-to-end (real LLM calls, real env vars). — **Reversibility:** reversible — isolated test file, no production code impact.

- **D-04:** Use each agent's configured port (9001–9010, from `a2a_ports()` in `config.py`). Tests add a check at the start: if the port is already occupied (e.g., `start_agents.sh` is running), the test is skipped with a descriptive message rather than failing. — **Reversibility:** reversible — per-test guard only.

- **D-05:** All 10 agent HTTP tests are marked `@pytest.mark.integration`. Default `pytest` run (no `-m` flag) excludes them; opt-in via `pytest -m integration`. Requires env vars (`TAVILY_API_KEY`, `OPENAI_API_KEY`, etc.) to be present. — **Reversibility:** reversible — marker only.

### Graph Integration Tests (A2A-07b)

- **D-06:** New file `tests/factcheck_agents/test_a2a_integration.py` — do NOT modify `test_debate_pipeline_integration.py` (which remains mocked for fast regression coverage). The new file contains a session-scoped pytest fixture that starts graph-path agents only. — **Reversibility:** reversible — new file, no changes to existing tests.

- **D-07:** Session-scoped fixture starts the 8 graph-path agents (`real_source=9003`, `fake_source=9004`, `social_loop=9005`, `agreement_gate=9006`, `real_advocate=9007`, `fake_advocate=9008`, `judge=9009`, `conclusion=9010`) as programmatic `uvicorn.Server` instances in background threads; polls `/.well-known/agent.json` for readiness with a timeout; tears down on session exit. `search_agent` (9001) and `evaluate_agent` (9002) are excluded — `search_agent` is on the M1 graph path not tested here; `evaluate_agent` is unused by both graphs. — **Reversibility:** reversible — fixture in test file only.

- **D-08:** New integration tests use shorter/simpler Vietnamese claims (not the Worldcup/NEI claims from the existing mocked suite) to stay within the < 60 s budget given real LLM call overhead. Claims should complete in under 30 s each on developer hardware. — **Reversibility:** reversible.

- **D-09:** `test_a2a_integration.py` tests are marked `@pytest.mark.integration` (same gate as D-05). — **Reversibility:** reversible.

### Backward Compat Fixes (A2A-08)

- **D-10:** Fix `run_fact_check()` in `factcheck_agents/__init__.py` and the invoke call in `mcp_server.py`: each generates `str(uuid.uuid4())` as `thread_id` and passes `config={"configurable": {"thread_id": thread_id}}` to `.invoke()`. Same pattern as the Phase 4 `cli.py` fix. External API unchanged. — **Reversibility:** reversible — internal impl only, no signature change.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope & Requirements
- `.planning/ROADMAP.md` §"Phase 5: Demo App + Tests" — success criteria (5 bullets), plan breakdown (05-01, 05-02), dependency on Phase 4
- `.planning/REQUIREMENTS.md` — A2A-06, A2A-06b (SSE bridge), A2A-07, A2A-07b (tests), A2A-08 (backward compat)
- `.planning/STATE.md` — port assignments (9001–9010), tunable defaults
- `.planning/PROJECT.md` — v3.1 milestone definition, out-of-scope list (React frontend unchanged)

### Phase 4 Decisions (locked — do not re-litigate)
- `.planning/phases/04-langgraph-a2a-client-wiring/04-CONTEXT.md` — D-01 (sync httpx bridge), D-02 (state diff shape), D-04 (@degrade_on_unavailable), D-05 (partial debate), D-06 (timeouts)
- `.planning/phases/04-langgraph-a2a-client-wiring/deferred-items.md` — thread_id crash in run_fact_check() / mcp_server.py (this phase fixes both)

### Existing Code (MUST read before writing streaming.py changes or test fixtures)
- `demo_app/backend/streaming.py` — full file; run_graph() thread architecture (D-02/D-03 pattern from v3.0 Phase 2), existing `except Exception as exc → {"type": "error"}` path to update, `done.is_set()` signal mechanism, NODE_STAGE_MAP
- `factcheck_agents/a2a_client.py` — `@degrade_on_unavailable`, per-agent degrade diffs (all include `"messages": [("assistant", "[AgentName] agent unavailable — degraded")]`), `AgentUnavailableError`
- `factcheck_agents/a2a_server.py` — `create_app(handler, cfg)` (returns FastAPI app for TestClient or uvicorn), `run_server()`, `BaseTaskHandler.agent_fn()` interface
- `factcheck_agents/__init__.py` — `run_fact_check()` at line 36: the invoke call without thread_id (D-10 fix target)
- `factcheck_agents/mcp_server.py` — the invoke call without thread_id (D-10 fix target)
- `factcheck_agents/config.py` — `a2a_ports()` mapping (agent name → port), used by test port allocation in D-04
- `factcheck_agents/agents/search_agent.py` — example TaskHandler structure; shows `SearchAgentHandler(BaseTaskHandler)`, `agent_fn()` signature, `run_server()` entry point
- `scripts/start_agents.sh` — reference for which agents run on which ports (validation cross-check)
- `tests/factcheck_agents/test_debate_pipeline_integration.py` — existing mocked integration tests (DO NOT MODIFY per D-06)
- `tests/factcheck_agents/test_a2a_client.py` — existing A2A client tests pattern (Phase 4 validation)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `create_app(handler, cfg)` in `a2a_server.py` — returns FastAPI app; use directly in test fixtures to instantiate each agent app without code changes
- `a2a_ports()` in `config.py` — canonical port mapping for test fixture port allocation and conflict-check in D-04
- `done.is_set()` + `_post()` pattern in `streaming.py` — already supports early close; D-02 just sets `done` from the detection path
- `uvicorn.Server` + `uvicorn.Config` — available in venv (uvicorn is a dep of a2a-sdk); use for programmatic startup in D-07

### Established Patterns
- `@pytest.mark.integration` + `pytest -m 'not integration'` — standard project test gating (TESTING.md); Phase 5 adopts this for all new A2A server tests
- `run_graph()` in `streaming.py` runs in a `ThreadPoolExecutor` thread; `_post()` bridges sync→async via `loop.call_soon_threadsafe`. The stage_error check in D-01/D-02 lives inside this thread.
- Degrade diffs always include `"messages": [("assistant", "...unavailable...")]` tuples — the scan target for D-01
- `cli.py` thread_id fix (Phase 4 deviation): `config={"configurable": {"thread_id": str(uuid.uuid4())}}` passed to `.invoke()` — copy this exact pattern for D-10

### Integration Points
- `streaming.py` `run_graph()` inner loop: add a post-`accumulated.update(node_output)` check for "unavailable" in messages, emit stage_error, set done
- `factcheck_agents/__init__.py:36` and `mcp_server.py:44` — the two `graph.invoke(state)` / `graph.invoke(state, config=...)` calls to patch with thread_id config
- New `test_a2a_integration.py` conftest session fixture: imports `create_app` from `a2a_server`, imports handler classes from each agent module, starts 8 uvicorn threads, yields, shuts down

</code_context>

<specifics>
## Specific Ideas

- For D-01 scan: check `any("unavailable" in str(part) for msg in node_output.get("messages", []) for part in (msg if isinstance(msg, (list, tuple)) else [msg]))` — covers both tuple `("assistant", "...")` and plain string message formats.
- The `stage_error` event shape: `{"type": "stage_error", "data": {"message": "Một số dịch vụ tạm thời không khả dụng. Không thể hoàn thành phân tích."}}` — posted via `_post()` then `done.set()`.
- For uvicorn-in-thread fixture (D-07): pattern is `config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"); server = uvicorn.Server(config); thread = threading.Thread(target=server.run, daemon=True); thread.start()` then poll `GET /.well-known/agent.json` in a loop with `time.sleep(0.1)` up to 10 s.
- For D-04 port-conflict skip: `import socket; s = socket.socket(); result = s.connect_ex(("127.0.0.1", port)); s.close()` — if `result == 0`, port is occupied; `pytest.skip(f"Port {port} already in use")`.
- `test_a2a_integration.py` minimal claim examples: `"Hà Nội là thủ đô của Việt Nam"` (factually true, fast evidence) or `"Mặt trăng làm từ phô mai"` (Moon is made of cheese — easily debunked, short debate).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 5-Demo App + Tests*
*Context gathered: 2026-08-18*
