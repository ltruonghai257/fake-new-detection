# Phase 2: Search / Evidence Agent - Context

**Gathered:** 2026-07-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Rewrite `factcheck_agents/agents/search_agent.py` to run 3 tier-separated query passes (trusted-domain, flagged-domain, unrestricted), tag every result with `source_tier` via `classify_domain()`, merge and cap the evidence list, build the `EvidenceGraph`, and write both `state["evidence_graph"]` and `state["evidence"]` (flat list for backward compat). Update `factcheck_agents/tools/web_search.py` to support `include_domains` domain filtering. No agent logic beyond search — no model calls, no verdict logic.

</domain>

<decisions>
## Implementation Decisions

### Query Pass Structure
- **D-01:** Run **3 passes** for every check: (1) `include_domains=trusted_list`, (2) `include_domains=flagged_list`, (3) unrestricted (no domain filter).
- **D-02:** Draft queries **once** from the statement (1 LLM call or heuristic fallback) and reuse the same queries across all 3 passes — only the domain scope differs.
- **D-03:** Tag every result (from any pass) by running `classify_domain(url)` — source tier reflects the actual source URL, not which pass retrieved it. Dedup by URL; first occurrence wins.

### Evidence Budget
- **D-04:** Each pass fetches up to `settings.max_results` items independently.
- **D-05:** After all 3 passes: merge + dedup by URL, sort by tier priority (trusted → flagged → unknown) then by score descending within each tier, cap the final list at **`2 × settings.max_results`** before building the graph.
- **D-06:** The tier-priority capped list is what goes into `EvidenceGraph.build_from_evidence()` and into `state["evidence"]`.

### `web_search()` Domain Filter API
- **D-07:** Add `include_domains: list[str] | None = None` parameter to `web_search()`, `_search_tavily()`, and `_search_google_cse()`. No `exclude_domains` needed for this phase.
- **D-08:** For Tavily: pass `include_domains` directly in the API request body as `"include_domains": include_domains`.
- **D-09:** For Google CSE: append the site-filter to the query string — `f"{query} {' OR '.join(f'site:{d}' for d in include_domains)}"`.

### Backward Compatibility
- **D-10:** `search_agent` continues to write **both** `state["evidence"]` (the tier-priority-capped flat list) and `state["evidence_graph"]`. Existing `evaluate_agent` and `conclusion_agent` continue to read `state["evidence"]` without modification until Phase 3/5 replace them.

### Claude's Discretion
- Same query content used across all 3 passes (D-02): query phrasing doesn't change based on tier — only search scope does. More LLM calls per tier would add cost for negligible quality gain.
- Vietnamese query forcing (SEARCH-02): the existing `SEARCH_QUERY_PROMPT` should be updated to explicitly instruct Vietnamese queries. Since this pipeline is exclusively Vietnamese fact-checking, always-Vietnamese is correct — no language detection library needed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Scope
- `.planning/REQUIREMENTS.md` §Search / Evidence Agent (SEARCH-01, SEARCH-02, SEARCH-03) — exact domain-filter, Vietnamese query, and graph-write requirements
- `.planning/REQUIREMENTS.md` §Source Tier Configuration (CONFIG-01, CONFIG-02, CONFIG-03) — env var names and defaults used by `classify_domain()`

### Phase 1 Context (locked decisions this phase depends on)
- `.planning/phases/01-state-config-evidence-graph-foundation/01-CONTEXT.md` — D-03 (`EvidenceGraph.build_from_evidence()` API), D-07 (`source_tier` Literal types), D-08 (`classify_domain()` reads from `settings`)

### Existing Code to Modify
- `factcheck_agents/agents/search_agent.py` — rewrite the `search_agent()` function; 3-pass loop replaces existing single-pass loop
- `factcheck_agents/tools/web_search.py` — add `include_domains` param to `web_search()`, `_search_tavily()`, `_search_google_cse()`
- `factcheck_agents/prompts.py` — update `SEARCH_QUERY_PROMPT` to instruct Vietnamese query generation (SEARCH-02)

### Project Constraints
- `.planning/PROJECT.md` §Constraints — work only inside `factcheck_agents/` and `tests/`; no new paid APIs; no new deps without confirmation
- `.planning/PROJECT.md` §Out of Scope — no paid X/Meta API

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `factcheck_agents/agents/search_agent.py` `_draft_queries()`: existing LLM-or-heuristic query drafting is reusable — just call it once and reuse queries across all 3 passes.
- `factcheck_agents/tools/web_search.py` `web_search()`: extend in-place by adding `include_domains` param; existing Tavily/CSE fallback logic stays intact.
- `factcheck_agents/graph_utils.py` `EvidenceGraph.build_from_evidence()` (Phase 1): classmethod ready to consume the final tagged evidence list.
- `factcheck_agents/source_tier.py` `classify_domain()` (Phase 1): pure function, apply to every result URL after retrieval.

### Established Patterns
- Dedup by URL with `seen: set` is already in `search_agent.py` — extend to track which URL was seen first (first-occurrence wins for tier tagging).
- `settings.max_results` is the single global results knob; cap at `2 × settings.max_results` keeps evidence bounded without a new config field.
- `total=False` on `Evidence` TypedDict — `source_tier` is optional and won't break existing callers that don't set it.

### Integration Points
- `search_agent()` return dict gains `"evidence_graph"` key alongside existing `"evidence"`, `"search_queries"`, `"messages"`.
- `factcheck_agents/state.py` `FactCheckState["evidence_graph"]` (Phase 1) is the write target.
- Phase 3 `verify_agent` will read `state["evidence_graph"]`; Phase 5 `conclusion_agent` will also read it. Until those phases land, existing agents read `state["evidence"]` unchanged.

</code_context>

<specifics>
## Specific Ideas

- Trusted domains default (CONFIG-01): `["vnexpress.net", "thanhnien.vn", "dantri.com.vn", "tuoitre.vn"]`
- Flagged domains default (CONFIG-02): `["kenh14.vn"]`
- Tier-priority sort key: `{"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}` — social after flagged but before unknown (consistent with the Literal order from Phase 1 D-07)
- Vietnamese query prompt update (SEARCH-02): add to `SEARCH_QUERY_PROMPT` an explicit instruction such as `"Generate all queries in Vietnamese."` — no langdetect import needed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 2-Search / Evidence Agent*
*Context gathered: 2026-07-19*
