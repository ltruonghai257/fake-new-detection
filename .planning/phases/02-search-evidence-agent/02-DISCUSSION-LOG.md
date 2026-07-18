# Phase 2: Search / Evidence Agent - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-19
**Phase:** 02-Search / Evidence Agent
**Areas discussed:** Query pass structure, Evidence budget, web_search() domain filter API, state['evidence'] backward compat

---

## Query Pass Structure

| Option | Description | Selected |
|--------|-------------|----------|
| 2 passes: trusted + flagged | Domain-restricted to trusted_list then flagged_list only. Misses unlisted sources. | |
| 3 passes: trusted + flagged + unrestricted | Add third open-web pass for broader coverage | ✓ |
| 2 passes + unrestricted fallback | Open-web only if results below threshold | |

**User's choice:** 3 passes: trusted + flagged + unrestricted

---

| Option | Description | Selected |
|--------|-------------|----------|
| Same queries for all 3 passes | 1 LLM call, domain scope is the only differentiator | ✓ |
| Separate query drafting per tier | 3 LLM calls, marginal quality gain | |

**User's choice:** Claude's discretion — same queries

---

| Option | Description | Selected |
|--------|-------------|----------|
| `classify_domain()` for all results | Source tier reflects actual URL domain regardless of pass | ✓ |
| `"unknown"` forced for unrestricted pass | Simpler but incorrect when trusted URLs appear in open-web results | |

**User's choice:** Claude's recommendation accepted — `classify_domain()` always

---

## Evidence Budget

| Option | Description | Selected |
|--------|-------------|----------|
| Per tier (3× total before dedup) | Each pass up to max_results; dedup handles reduction | initial preference |
| Shared budget split across tiers | Proportional allocation; slots may go unfilled | |
| Per-tier cap + global dedup cap | Two config knobs | |
| Per-tier + tier-priority sort + 2× cap | Each tier gets max_results; merged list sorted by tier priority, capped at 2× | ✓ |

**User's choice:** Per-tier fetch → merge+dedup → tier-priority sort (trusted→flagged→unknown, then by score) → cap at 2×max_results
**Notes:** User's concern was graph size and downstream PhoBERT/LLM cost. Final recommendation addressed this with tier-priority sort and 2× global cap.

---

## web_search() Domain Filter API

| Option | Description | Selected |
|--------|-------------|----------|
| `include_domains` only | Add one param; Tavily native, CSE appends `site:` | ✓ |
| Add `exclude_domains` too | Future-proofing for Phase 4 | |
| Different approach | — | |

**User's choice:** Accept — `include_domains` only

---

| Option | Description | Selected |
|--------|-------------|----------|
| Prepend `site:` filter | `"site:d1 OR site:d2 <query>"` | |
| Append `site:` filter | `"<query> site:d1 OR site:d2"` | ✓ |

**User's choice:** Append

---

## state['evidence'] Backward Compat

| Option | Description | Selected |
|--------|-------------|----------|
| Keep both | Write `state["evidence"]` + `state["evidence_graph"]`; existing agents unaffected | ✓ |
| Graph only | Drop flat list; breaks evaluate_agent/conclusion_agent until Phase 3/5 | |

**User's choice:** Keep both

---

## Claude's Discretion

- **Same queries across all passes:** query phrasing doesn't change based on tier — domain scope is the differentiator. 1 LLM call is correct.
- **Vietnamese query forcing (SEARCH-02):** always-Vietnamese instruction added to `SEARCH_QUERY_PROMPT` without langdetect — pipeline is exclusively Vietnamese fact-checking.

## Deferred Ideas

None.
