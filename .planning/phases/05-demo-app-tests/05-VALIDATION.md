---
phase: 5
slug: demo-app-tests
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-22
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property               | Value                                                 |
| ---------------------- | ----------------------------------------------------- |
| **Framework**          | pytest 9.1.1                                          |
| **Config file**        | `pytest.ini`                                          |
| **Quick run command**  | `pytest tests/ -m "not integration" -x -q`            |
| **Full suite command** | `pytest tests/ -m "not integration"`                  |
| **Integration suite**  | `pytest tests/ -m integration -v` (requires env vars) |
| **Estimated runtime**  | ~10 s (non-integration) · ~60 s (integration)         |

---

## Sampling Rate

-   **After every task commit:** Run `pytest tests/ -m "not integration" -x -q`
-   **After every plan wave:** Run `pytest tests/ -m "not integration"`
-   **Before `/gsd-verify-work`:** Full non-integration suite must be green; integration suite run manually with env vars
-   **Max feedback latency:** 15 seconds (non-integration)

---

## Per-Task Verification Map

| Task ID  | Plan | Wave | Requirement | Threat Ref | Secure Behavior                                                | Test Type   | Automated Command                                                         | File Exists | Status     |
| -------- | ---- | ---- | ----------- | ---------- | -------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------- | ----------- | ---------- |
| 05-01-01 | 01   | 1    | A2A-06b     | —          | stage_error emitted on agent unavailability; stream closes 200 | unit        | `pytest tests/ -m "not integration" -k "streaming" -q`                    | ❌ W0       | ⬜ pending |
| 05-01-02 | 01   | 1    | A2A-06b     | —          | done.set() called before break; no 500 errors                  | unit        | `pytest tests/ -m "not integration" -k "streaming" -q`                    | ❌ W0       | ⬜ pending |
| 05-02-01 | 02   | 2    | A2A-07      | —          | all 10 TaskHandler HTTP tests pass                             | integration | `pytest tests/factcheck_agents/test_agent_http.py -m integration -v`      | ❌ W0       | ⬜ pending |
| 05-02-02 | 02   | 2    | A2A-07b     | —          | graph integration tests pass with session fixture; < 60 s      | integration | `pytest tests/factcheck_agents/test_a2a_integration.py -m integration -v` | ❌ W0       | ⬜ pending |
| 05-02-03 | 02   | 2    | A2A-08      | —          | run_fact_check() passes thread_id; no crash                    | unit        | `pytest tests/ -m "not integration" -k "entrypoint or run_fact_check" -q` | ❌ W0       | ⬜ pending |
| 05-02-04 | 02   | 2    | A2A-08      | —          | mcp_server.py passes thread_id; no crash                       | unit        | `pytest tests/ -m "not integration" -k "mcp" -q`                          | ❌ W0       | ⬜ pending |
| 05-02-05 | 02   | 2    | A2A-06      | —          | full non-integration suite green (backward compat)             | regression  | `pytest tests/ -m "not integration" -q`                                   | ✅          | ⬜ pending |

_Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky_

---

## Wave 0 Requirements

-   [ ] `tests/demo_app/test_streaming.py` — stubs for A2A-06b stage_error detection (or inline in plan 05-01)
-   [ ] `tests/factcheck_agents/test_agent_http.py` — stubs for A2A-07 (new file)
-   [ ] `tests/factcheck_agents/test_a2a_integration.py` — stubs for A2A-07b (new file)
-   [ ] `tests/factcheck_agents/test_entrypoints.py` — stubs for A2A-08 thread_id fix (new file)

_Existing framework (pytest + pytest.ini) covers all phase requirements — no new framework install needed._

---

## Manual-Only Verifications

| Behavior                                                 | Requirement | Why Manual                                 | Test Instructions                                                                                                                                                     |
| -------------------------------------------------------- | ----------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SSE `stage_error` appears in browser and closes cleanly  | A2A-06b     | Requires live browser + real agent servers | 1. Start demo app. 2. Stop one A2A agent. 3. Submit claim. 4. Verify SSE stream emits `stage_error` event and page shows Vietnamese error message. HTTP 200 (no 500). |
| Demo streaming still works end-to-end with all agents up | A2A-06      | Full browser UX                            | 1. Start all agents via `scripts/start_agents.sh`. 2. Submit real claim in browser. 3. Verify debate animation, verdict, evidence display.                            |

---

## Validation Sign-Off

-   [ ] All tasks have `<automated>` verify or Wave 0 dependencies
-   [ ] Sampling continuity: no 3 consecutive tasks without automated verify
-   [ ] Wave 0 covers all MISSING references
-   [ ] No watch-mode flags
-   [ ] Feedback latency < 15s
-   [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
