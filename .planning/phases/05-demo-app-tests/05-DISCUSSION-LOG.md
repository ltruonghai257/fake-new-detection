# Phase 5: Demo App + Tests - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-18
**Phase:** 05-demo-app-tests
**Areas discussed:** stage_error detection, Agent HTTP test approach, Integration fixture scope, Deferred thread_id fix

---

## stage_error detection (A2A-06b)

| Option | Description | Selected |
|--------|-------------|----------|
| Scan messages field | Check node_output.get("messages", []) for "unavailable" strings — degrade diffs already inject these | ✓ |
| Add degraded_agents state flag | New FactCheckState field populated by degrade decorator — requires touching a2a_client.py and state.py | |
| Check verdict UNVERIFIED only | Emit stage_error only on final UNVERIFIED verdict — coarse, doesn't show which stage failed | |

**User's choice:** Scan messages field

**Follow-up — what stage_error does:**

| Option | Description | Selected |
|--------|-------------|----------|
| Emit + continue stream | Post stage_error then keep streaming toward a (possibly degraded) verdict | |
| Emit + close immediately | Post stage_error then set done.is_set() — stream closes, HTTP 200 | ✓ |

**User's choice:** Emit + close immediately

**Follow-up — Vietnamese message content:**

| Option | Description | Selected |
|--------|-------------|----------|
| Generic fixed string | Same message for all agents: "Một số dịch vụ tạm thời không khả dụng…" | |
| Per-agent from degrade message | Map English degrade message to Vietnamese per agent — more informative | |
| You decide | Claude picks simplest approach satisfying A2A-06b | ✓ |

**Notes:** Claude chose generic fixed string — simpler, no lookup map.

---

## Agent HTTP test approach (A2A-07)

| Option | Description | Selected |
|--------|-------------|----------|
| TestClient (simplest) | TestClient(create_app(handler, cfg)) — no new deps, no real port, sync | |
| uvicorn thread + httpx | Real uvicorn.Server in background thread, httpx.Client to localhost:PORT | ✓ |
| Install pytest-anyio | Add anyio + pytest-anyio, use AsyncClient(app=...) transport | |

**User's choice:** uvicorn thread + httpx

**Follow-up — test fidelity:**

| Option | Description | Selected |
|--------|-------------|----------|
| Real Task + mock agent fn | Real A2A HTTP through full stack; underlying agent fn mocked with fixtures | |
| Real Task + real agent fn | True end-to-end — A2A HTTP + real agent logic, requires env vars | ✓ |

**User's choice:** Real Task + real agent fn

**Follow-up — port strategy:**

| Option | Description | Selected |
|--------|-------------|----------|
| Use real ports (9001–9010) | Fail (skip) if port already occupied; conflict check at test start | ✓ |
| Ephemeral ports (OS assigned) | Bind to port 0; patch config per test — no conflicts ever | |

**User's choice:** Use real ports (9001–9010), with skip on conflict

**Follow-up — test marker:**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, integration marker | @pytest.mark.integration, excluded by default | ✓ |
| No marker, always run | Always runs, requires credentials in CI | |

**User's choice:** @pytest.mark.integration

---

## Integration fixture scope (A2A-07b)

| Option | Description | Selected |
|--------|-------------|----------|
| Update existing tests | Remove mocks from test_debate_pipeline_integration.py; add server fixture | |
| New file, keep existing | New test_a2a_integration.py; existing tests untouched | |
| You decide | Claude picks approach with least risk | ✓ |

**Notes:** Claude chose new file — avoids touching the 5 pre-existing failures in test_debate_pipeline_integration.py.

**Follow-up — fixture server startup:**

| Option | Description | Selected |
|--------|-------------|----------|
| Use start_agents.sh | Fixture calls existing shell script | |
| Programmatic uvicorn threads | 8 uvicorn.Server instances in background threads | ✓ |
| subprocess.Popen per agent | One process per agent — most isolated, slower | |

**User's choice:** Programmatic uvicorn threads

**Follow-up — which agents to start:**

| Option | Description | Selected |
|--------|-------------|----------|
| All 10 agents | Start all regardless of which tests run | |
| Only graph-path agents | Start only the 8 agents called by build_debate_graph() | ✓ |

**User's choice:** Only graph-path agents (8 of 10)

**Follow-up — test claims:**

| Option | Description | Selected |
|--------|-------------|----------|
| Same 2 claims (Worldcup + NEI) | Port existing claims from mocked suite | |
| New minimal claims | Shorter/simpler claims to stay under 60 s budget | ✓ |

**User's choice:** New minimal claims

---

## Deferred thread_id fix (A2A-08)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — fix both in Phase 5 | run_fact_check() and mcp_server.py each get uuid4 thread_id | ✓ |
| No — separate maintenance | Defer, A2A-08 verification partial | |

**User's choice:** Fix both in Phase 5

**Follow-up — fix approach:**

| Option | Description | Selected |
|--------|-------------|----------|
| uuid4 per call | Minimal: generate str(uuid.uuid4()) in each invoke call | ✓ |
| Shared helper | Extract _make_thread_config() helper — DRY but more indirection | |

**User's choice:** uuid4 per call (same pattern as cli.py Phase 4 fix)

---

## Claude's Discretion

- **stage_error Vietnamese message:** generic fixed string `"Một số dịch vụ tạm thời không khả dụng. Không thể hoàn thành phân tích."` — no per-agent lookup map
- **Integration test scope:** new `test_a2a_integration.py` file (not modifying existing mocked tests) — avoids the 5 pre-existing failures in `test_debate_pipeline_integration.py`

## Deferred Ideas

None — discussion stayed within phase scope.
