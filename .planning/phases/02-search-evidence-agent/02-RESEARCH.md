# Phase 2: Search / Evidence Agent — Research

**Phase:** 02 — Search / Evidence Agent
**Requirements:** SEARCH-01, SEARCH-02, SEARCH-03
**Researched:** 2026-07-19

---

## What exists today

### `search_agent.py` (63 lines)
Single-pass search: draft queries once → iterate over `web_search(q)` for each query → dedup by URL → sort by score → cap at `settings.max_results`. Returns `{"search_queries", "evidence", "messages"}`. No `source_tier` tagging, no evidence graph, no tier-separated passes.

### `web_search.py` (91 lines)
`web_search(query, max_results)` → tries Tavily then Google CSE. **No `include_domains` support anywhere.** Tavily call passes only `api_key`, `query`, `max_results`, `search_depth`, `include_answer`. Google CSE appends nothing beyond the raw query.

### `prompts.py` — `SEARCH_QUERY_PROMPT`
Says `"Prefer the claim's original language."` — needs to change to explicitly instruct Vietnamese.

### Phase 1 outputs (already in codebase)
- `graph_utils.py` `EvidenceGraph.build_from_evidence(evidence_list)` — classmethod, reads `item.get("source_tier", "unknown")` from each dict. **Ready to consume** the tagged list.
- `source_tier.py` `classify_domain(url)` — pure function, reads `settings.trusted_domains` / `settings.flagged_domains`. Returns `"trusted"` | `"flagged"` | `"unknown"`.
- `state.py` `Evidence.source_tier: Literal["trusted", "flagged", "social", "unknown"]` — already optional (`total=False`), so untagged callers won't break.
- `state.py` `FactCheckState.evidence_graph: Optional[Any]` — write target already present.
- `config.py` `settings.trusted_domains`, `settings.flagged_domains` — already parsed from env.

---

## Implementation approach

### Plan 02-01 — `search_agent.py` rewrite

The main change: replace the single-pass loop with a 3-pass loop and add tier tagging + graph construction.

**`_draft_queries()` is fully reusable** — call it once, pass the same queries to all 3 passes.

**3-pass structure:**
```
Pass 1: web_search(q, include_domains=trusted_list)   → tag results "trusted" (then re-check with classify_domain)
Pass 2: web_search(q, include_domains=flagged_list)   → tag results "flagged"
Pass 3: web_search(q)  # unrestricted                 → classify_domain per URL
```

**Tier tagging per result**: Always run `classify_domain(url)` on the actual result URL — the tier reflects the actual domain, not which pass fetched it (D-03 from CONTEXT.md).

**Dedup logic**: Keep a `seen: set` of URLs; first occurrence wins. This matches the existing pattern already in `search_agent.py`.

**Post-merge sort + cap**: Sort by `tier_priority[source_tier]` (trusted=0, flagged=1, social=2, unknown=3) then by `score` descending within tier. Cap at `2 × settings.max_results`.

**Return dict gains `"evidence_graph"` key**: Call `EvidenceGraph.build_from_evidence(evidence)` on the final capped list. Return both `"evidence"` (backward compat for phases 3–5 that read `state["evidence"]`) and `"evidence_graph"`.

**`trusted_domains` / `flagged_domains` as lists**: `[d.strip() for d in settings.trusted_domains.split(",") if d.strip()]` — same pattern `classify_domain()` already uses.

### Plan 02-02 — `web_search.py` + `prompts.py`

**`web_search.py`**: Add `include_domains: list[str] | None = None` to `web_search()`, `_search_tavily()`, `_search_google_cse()`.

- **Tavily**: Pass `"include_domains": include_domains` in the request body only when `include_domains` is not None/empty. Tavily API accepts this field natively.
- **Google CSE**: Append site-restriction to query string: `query + " " + " OR ".join(f"site:{d}" for d in include_domains)` when `include_domains` is non-empty. Native CSE has `siteSearch` param but the query-string approach is simpler and consistent with existing pattern.

**`prompts.py`**: Update `SEARCH_QUERY_PROMPT` — change `"Prefer the claim's original language."` to `"Generate all queries in Vietnamese."`. This is always correct since the pipeline is exclusively Vietnamese fact-checking (SEARCH-02, Claude's Discretion from CONTEXT.md).

---

## Risk surface

| Risk | Mitigation |
|------|-----------|
| Tavily returns fewer results when `include_domains` is very narrow | 3rd pass (unrestricted) always fills the gap; cap is `2×max_results` not `max_results` |
| Google CSE `num=10` cap unchanged when site filters narrow results | Accepted; unrestricted pass compensates |
| `classify_domain()` parses settings on every call (list rebuild) | Negligible at this scale; no caching needed |
| Passing empty `include_domains=[]` to Tavily | Guard: only include `include_domains` in payload when list is non-empty |

---

## Validation Architecture

Tests needed (Phase 8, but design now):

1. **`test_search_agent_tiers.py`** (or inline in existing test file):
   - Mock `web_search` to return results with known URLs; verify `source_tier` on each evidence item after agent runs.
   - Verify `state["evidence_graph"]` is an `EvidenceGraph` instance with nodes.
   - Verify `state["evidence"]` still present (backward compat).
   - Verify dedup: two items with same URL — only first kept.
   - Verify cap: more than `2×max_results` raw results → capped.

2. **`test_web_search_domain_filter.py`**:
   - Mock `requests.post` / `requests.get`; assert `include_domains` appears in Tavily payload when non-empty.
   - Assert Google CSE query string includes `site:` tokens.
   - Assert `include_domains=None` → no domain filtering applied (existing behavior unchanged).

3. **`test_search_query_prompt_vietnamese.py`**:
   - Assert `SEARCH_QUERY_PROMPT` contains "Vietnamese".

---

## ## RESEARCH COMPLETE

Phase 2 is a focused rewrite of two files plus one prompt string. All Phase 1 outputs are in place and ready to consume. No new dependencies needed. Implementation can proceed directly to PLAN.md.
