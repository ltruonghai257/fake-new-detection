---
phase: 04
slug: langgraph-a2a-client-wiring
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-17
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1 (Python 3.12.12, `.venv`) |
| **Config file** | `pytest.ini` (repo root; `testpaths=tests`, `pythonpath=. src`) |
| **Quick run command** | `.venv/bin/python -m pytest tests/factcheck_agents/test_a2a_client.py tests/factcheck_agents/test_debate_node_partial.py tests/factcheck_agents/test_graph_wiring.py -q` |
| **Full suite command** | `.venv/bin/python -m pytest tests/factcheck_agents/ -q --tb=no` |
| **Estimated runtime** | ~6 s (phase tests) / ~8 min (full suite, includes slow integration tests) |

---

## Sampling Rate

- **After every task commit:** Run quick run command (~6 s)
- **After every plan wave:** Run full suite command (~8 min)
- **Before `/gsd-verify-work`:** Full suite must be green — baseline parity: 104 collected, 99 passed / 5 failed (pre-existing label-mapping drift, tracked in `deferred-items.md`, not caused by Phase 04)
- **Max feedback latency:** ~10 s for phase-relevant tests

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 04-01 | 1 | A2A-04 | D-01 | Per-agent timeouts default 120/60/30; global override wins | unit | `pytest test_a2a_client.py::test_settings_timeout_defaults test_a2a_client.py::test_timeout_for_per_agent_defaults` | ✅ | ✅ green |
| 04-01-02 | 04-01 | 1 | A2A-04 | T-01, D-04, D-01, D-02, S-01, S-02 | `AgentUnavailableError` on HTTPError/FAILED; WORKING → `{}`; loopback-only client; `A2A-Version` + `ROLE_USER` | unit | `pytest test_a2a_client.py::test_agent_unavailable_error_attributes_and_message test_a2a_client.py::test_call_agent_*` | ✅ | ✅ green |
| 04-01-03 | 04-01 | 1 | A2A-04 | D-03 | `degrade_on_unavailable` catches → per-agent diff; shallow-copy isolation | unit | `pytest test_a2a_client.py::test_degrade_on_unavailable_* test_a2a_client.py::test_wrapper_degrade_diffs` | ✅ | ✅ green |
| 04-01-04 | 04-01 | 1 | A2A-04 | T-01, T-02 | 10 typed wrappers; `search_agent` rebuilds EvidenceGraph from evidence list (repr discarded) | unit | `pytest test_a2a_client.py::test_search_agent_*` | ✅ | ✅ green |
| 04-01-05 | 04-01 | 1 | A2A-04 | — | All error paths importable & verified | unit | `pytest test_a2a_client.py` (full file, 16 tests) | ✅ | ✅ green |
| 04-02-01 | 04-02 | 2 | A2A-05 | S-02 | No direct imports of A2A-wrapped agents in graph.py; `route_after_agreement` stays local | unit | `pytest test_graph_wiring.py::test_graph_has_no_direct_agent_imports test_graph_wiring.py::test_graph_agreement_gate_import_binds_only_route test_graph_wiring.py::test_graph_imports_from_a2a_client` | ✅ | ✅ green |
| 04-02-02 | 04-02 | 2 | A2A-05b | D-03, T-01 | Partial debate: both-down → `agent_unavailable`; one-sided skips convergence; IN-07 no-None appends | unit | `pytest test_debate_node_partial.py` (6 tests) | ✅ | ✅ green |
| 04-02-03 | 04-02 | 2 | A2A-05 | — | Patch targets effective at graph call site (deviated: `graph.*` not `a2a_client.*`) | unit | `pytest test_graceful_degrade.py` | ✅ | ✅ green |
| 04-02-04 | 04-02 | 2 | A2A-05 | — | Debate pipeline runs with mocked agents (2 of 4 tests; worldcup/nei failures are pre-existing, deferred) | unit | `pytest test_debate_pipeline_integration.py` | ✅ | ⚠️ flaky (pre-existing 2/4) |
| 04-02-05 | 04-02 | 2 | A2A-05 | — | Full suite at baseline parity, zero regressions | unit | `pytest tests/factcheck_agents/ -q --tb=no` | ✅ | ✅ green |
| 04-02-06 | 04-02 | 2 | A2A-05 | — | `build_graph()`/`build_debate_graph()` compile; imports resolve | unit | `pytest test_graph_wiring.py::test_build_graph_compiles` | ✅ | ✅ green |
| 04-02-07 | 04-02 | 2 | A2A-05 | T-01, D-04, P-01 | Live end-to-end: servers up → valid verdict <60s; all down → UNVERIFIED/0.0, no crash | e2e-manual | — (see Manual-Only) | ❌ | ⬜ manual |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements — no Wave 0 needed. Phase 04 added 25 tests:
`test_a2a_client.py` (16), `test_debate_node_partial.py` (6), `test_graph_wiring.py` (+3 AST structure tests).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CLI end-to-end smoke test — all 10 A2A servers up: exit 0, valid verdict, <60 s | A2A-05 | Requires 10 live uvicorn agent servers + real LLM config; cannot run in CI. Verified live during execution (58 s verdict). | `scripts/start_agents.sh`; wait for all `/.well-known/agent.json` (ports 9001–9010); `python -m factcheck_agents.cli "Tin tức test"` → exit 0, verdict dict with valid label |
| CLI end-to-end smoke test — all servers down: exit 0, UNVERIFIED, confidence 0.0, no crash | A2A-05b | Requires stopping all servers; degrade path validated live. | `scripts/stop_agents.sh`; `python -m factcheck_agents.cli "Tin tức test"` → exit 0, `label == "UNVERIFIED"`, `confidence == 0.0`; restart with `scripts/start_agents.sh` |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (except 04-02-07, manual-only e2e, documented above)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none — infra pre-existed)
- [x] No watch-mode flags
- [x] Feedback latency < 10 s (phase tests)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-08-17

---

## Audit Notes (2026-08-17)

- **State B reconstruction** — no VALIDATION.md existed at execution time; per-task map reconstructed from 04-01/04-02 PLANs, SUMMARYs, and VERIFICATION.md, then gap-filled by the Nyquist auditor.
- **Gaps found: 3, resolved: 3** — G1: `test_a2a_client.py` (A2A-04, was inline-`python -c` only); G2: `test_debate_node_partial.py` (A2A-05b, partial-debate semantics never persisted); G3: AST wiring checks in `test_graph_wiring.py` (A2A-05).
- **Non-blocking auditor findings:** (1) `Settings.a2a_client_timeout` stores a string when env-set (no `int()` cast) — plan-conformant, runtime-safe via `float()` cast in `_timeout_for`; (2) `_timeout_for` raises `AttributeError` for unknown agent names — only the 10 known agents are used.
- **Pre-existing failures (5) untouched** — tracked in `deferred-items.md`; suite at exact baseline parity (99 pass / 5 fail).
