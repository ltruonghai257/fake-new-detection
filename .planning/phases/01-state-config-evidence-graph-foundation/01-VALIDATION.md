---
phase: 1
slug: state-config-evidence-graph-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-19
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pytest.ini` |
| **Quick run command** | `pytest tests/factcheck_agents/ -x -q` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/factcheck_agents/ -x -q`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01-01 | 1 | CONFIG-01, CONFIG-02 | — | N/A | unit | `pytest tests/factcheck_agents/ -x -q -k "source_tier or config"` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01-01 | 1 | EVGRAPH-01, EVGRAPH-02 | — | N/A | unit | `pytest tests/factcheck_agents/ -x -q -k "state"` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01-01 | 1 | CONFIG-03 | — | N/A | unit | `pytest tests/factcheck_agents/test_source_tier.py -x -q` | ❌ W0 | ⬜ pending |
| 01-02-01 | 01-02 | 2 | EVGRAPH-01 | — | N/A | unit | `pytest tests/factcheck_agents/test_evidence_graph.py -x -q` | ❌ W0 | ⬜ pending |
| 01-02-02 | 01-02 | 2 | TEST-01 | — | N/A | unit | `pytest tests/factcheck_agents/test_source_tier.py -v` | ❌ W0 | ⬜ pending |
| 01-02-03 | 01-02 | 2 | TEST-02 | — | N/A | unit | `pytest tests/factcheck_agents/test_evidence_graph.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/factcheck_agents/__init__.py` — create empty init to make it a package
- [ ] `tests/factcheck_agents/test_source_tier.py` — stubs for TEST-01
- [ ] `tests/factcheck_agents/test_evidence_graph.py` — stubs for TEST-02

*Wave 0 must create test directory and stub files before plan 01-02 tasks run.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `networkx` available after `uv sync --extra agents` | EVGRAPH-01 | Install step | Run `uv sync --extra agents && python -c "import networkx; print(networkx.__version__)"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
