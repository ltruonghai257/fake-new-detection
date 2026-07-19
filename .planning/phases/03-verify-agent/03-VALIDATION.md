---
phase: 3
slug: verify-agent
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-19
completed: 2026-07-19
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` |
| **Quick run command** | `pytest tests/factcheck_agents/test_verify_agent.py -q` |
| **Full suite command** | `pytest tests/ -q --ignore=tests/processing/coolant` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/factcheck_agents/test_verify_agent.py -q`
- **After every plan wave:** Run `pytest tests/ -q --ignore=tests/processing/coolant`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-T1 | 03-01 | 0 | VERIFY-01/02/03 | — | `test_verify_agent.py` stubs exist | unit | `pytest tests/factcheck_agents/test_verify_agent.py --collect-only` | ✅ W0 | ✅ green |
| 03-01-T2 | 03-01 | 1 | VERIFY-01/02/03 | — | `verify_agent.py` created with `ThreadPoolExecutor(max_workers=2)` and `_compute_reliability_signal` | unit | `pytest tests/factcheck_agents/test_verify_agent.py -q` | ✅ W1 | ✅ green |
| 03-01-T3 | 03-01 | 1 | VERIFY-01/02/03 | — | `verify_agent` exported from `agents/__init__.py` | unit | `python -c "from factcheck_agents.agents import verify_agent"` | ✅ W1 | ✅ green |
| 03-01-T4 | 03-01 | 1 | VERIFY-01/02/03 | — | Real tests for reliability_signal and no-crash degrade | unit | `pytest tests/factcheck_agents/test_verify_agent.py -v` | ✅ W1 | ✅ green |
| 03-01-T5 | 03-01 | 1 | VERIFY-01/02/03 | — | Full suite regression check | integration | `pytest tests/ -q --ignore=tests/processing/coolant` | ✅ W1 | ✅ green |
| 03-02-T1 | 03-02 | 1 | EVGRAPH-03 | — | `_TIER_ORDER` constant and tier-sort in `build_evidence_text()` | unit | `pytest tests/factcheck_agents/test_verify_agent.py -k "build_evidence_text" -q` | ✅ W1 | ✅ green |
| 03-02-T2 | 03-02 | 1 | EVGRAPH-03 | — | Tier-ordering tests pass | unit | `pytest tests/factcheck_agents/test_verify_agent.py -k "build_evidence_text" -v` | ✅ W1 | ✅ green |
| 03-02-T3 | 03-02 | 1 | EVGRAPH-03 | — | Full suite regression check | integration | `pytest tests/ -q --ignore=tests/processing/coolant` | ✅ W1 | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/factcheck_agents/test_verify_agent.py` — stubs for VERIFY-01, VERIFY-02, VERIFY-03 (signal computation, concurrent execution, no-crash degrade)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live PhoBERT + COOLANT concurrent inference latency | VERIFY-02 | Requires real checkpoints/GPU | Run CLI end-to-end and inspect `[Verify]` message shows both model results |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
