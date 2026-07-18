---
phase: 2
slug: search-evidence-agent
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-19
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 02-01 | 1 | SEARCH-01 | — | include_domains passed correctly to Tavily/CSE | unit | `pytest tests/ -k "test_web_search" -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 02-01 | 1 | SEARCH-01 | — | 3-pass structure runs, dedup by URL first-occurrence | unit | `pytest tests/ -k "test_search_agent" -q` | ❌ W0 | ⬜ pending |
| 02-01-03 | 02-01 | 1 | CONFIG-03/SEARCH-01 | — | source_tier set by classify_domain on actual URL | unit | `pytest tests/ -k "test_search_agent" -q` | ❌ W0 | ⬜ pending |
| 02-01-04 | 02-01 | 1 | SEARCH-03 | — | evidence_graph populated and is EvidenceGraph instance | unit | `pytest tests/ -k "test_search_agent" -q` | ❌ W0 | ⬜ pending |
| 02-01-05 | 02-01 | 1 | SEARCH-03 | — | state["evidence"] backward compat preserved | unit | `pytest tests/ -k "test_search_agent" -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02-02 | 2 | SEARCH-02 | — | SEARCH_QUERY_PROMPT contains "Vietnamese" | unit | `pytest tests/ -k "test_search_query_prompt" -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02-02 | 2 | SEARCH-01 | — | Tavily payload includes include_domains when non-empty | unit | `pytest tests/ -k "test_web_search" -q` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02-02 | 2 | SEARCH-01 | — | Google CSE query string includes site: tokens | unit | `pytest tests/ -k "test_web_search" -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/factcheck_agents/test_search_agent.py` — stubs for SEARCH-01, SEARCH-03 (3-pass, tier tagging, evidence_graph, backward compat)
- [ ] `tests/factcheck_agents/test_web_search.py` — stubs for SEARCH-01 (include_domains Tavily payload, Google CSE site: filter)
- [ ] `tests/factcheck_agents/test_prompts.py` — stub for SEARCH-02 (Vietnamese in prompt)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live Tavily call with include_domains returns correct tier-filtered results | SEARCH-01 | Requires real API key | Run CLI with `FACTCHECK_TRUSTED_DOMAINS=vnexpress.net` and verify evidence items from vnexpress.net are present |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
