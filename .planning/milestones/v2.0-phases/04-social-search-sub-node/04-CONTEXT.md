# Phase 4: Social Search Sub-Node - Context

**Gathered:** 2026-07-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Create `factcheck_agents/agents/social_search_agent.py` — a LangGraph node that, given `state["search_queries"]` already in state, runs site-restricted queries (`site:twitter.com OR site:facebook.com`) through the existing `web_search()` tool (max 3 results per query), tags each result `source_tier="social"`, fetches evidence images/captions via `_fetch_evidence_image()`, and merges new nodes + `"mentions"` edges into the existing `state["evidence_graph"]` DiGraph. Returns only `evidence_graph` and `messages` — no flat evidence list update, no model calls, no routing logic.

The node assumes it is only called when `reliability_signal=True`; the conditional edge is Phase 6's responsibility. `_fetch_evidence_image()` must be extracted from `search_agent.py` into a shared location so both agents can import it.

</domain>

<decisions>
## Implementation Decisions

### Query Source
- **D-01:** Reuse `state["search_queries"]` — the queries already drafted in Vietnamese by `search_agent` (via LLM or heuristic fallback). Loop: `for q in state["search_queries"]: web_search(q, max_results=3, include_domains=["twitter.com", "facebook.com"])`. No extra LLM call. `search_queries` is always non-empty (heuristic fallback guarantees at least the raw statement as a query).

### State Write-Back
- **D-02:** **Graph only** — `social_search_agent` returns `{"evidence_graph": updated_graph, "messages": [...]}`. Does **not** write to `state["evidence"]` (flat list). Phase 5 `conclusion_agent` reads `evidence_graph` directly. The flat list remains as the search_agent left it.

### Social Node Edge Type
- **D-03:** All new social evidence nodes connect to the statement node with edge type `"mentions"`. Social posts have not been scored by PhoBERT or COOLANT — asserting `supports`/`contradicts` would be ungrounded. Phase 5 conflict detection reads `contradicts` edges; `"mentions"` edges are safely neutral.

### Image Fetching
- **D-04:** Mirror `search_agent` — call `_fetch_evidence_image(url)` on each social result URL and assign `image_path` / `image_caption` to the evidence item before adding it to the graph. The function already handles failures silently (`return None, None` on any exception). Note: COOLANT has already run by the time this node executes (Phase 3), so these images are stored as graph metadata only — not fed to COOLANT in this phase.

### _fetch_evidence_image() Refactor
- **D-05:** Extract `_fetch_evidence_image()` from `factcheck_agents/agents/search_agent.py` into a shared module (`factcheck_agents/helpers.py` or similar). Update `search_agent.py` to import from the new location. `social_search_agent.py` imports from the same shared module. Keeps both agents DRY without duplication.

### Claude's Discretion
- Dedup strategy: maintain a `seen: set` of URLs from the existing evidence_graph node IDs before looping queries. Skip any social result whose URL is already in `seen`. Same first-occurrence-wins pattern as `search_agent`.
- No-results behavior: if all web_search calls return empty (provider down, domains blocked), return unchanged `evidence_graph` and a diagnostic message. Consistent with graceful-degrade pattern throughout.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Scope
- `.planning/REQUIREMENTS.md` §Social Search Sub-Node (SOCIAL-01, SOCIAL-02, SOCIAL-03) — exact domain restrictions, max 3 results, merge contract, no internal routing check
- `.planning/PROJECT.md` §Constraints — scope boundary (factcheck_agents/ and tests/ only); no new paid APIs; no new deps without confirmation

### Prior Phase Context (locked decisions this phase depends on)
- `.planning/phases/01-state-config-evidence-graph-foundation/01-CONTEXT.md` — D-03 (`EvidenceGraph` 4-method API), D-05 (node/edge structure: statement node + evidence nodes, edge types `supports`/`contradicts`/`mentions`)
- `.planning/phases/02-search-evidence-agent/02-CONTEXT.md` — D-10 (backward compat: `state["evidence"]` flat list stays as written by search_agent; Phase 4 does not touch it)
- `.planning/phases/03-verify-agent/03-CONTEXT.md` — D-06/D-07 (tier sort order including `"social": 2`); confirms social_search runs after verify_agent (COOLANT already executed)

### Existing Code to Modify / Create
- `factcheck_agents/agents/social_search_agent.py` — new file
- `factcheck_agents/agents/search_agent.py` — remove `_fetch_evidence_image()`, import from shared helper (D-05)
- `factcheck_agents/helpers.py` (or equivalent shared module) — new file containing `_fetch_evidence_image()`
- `factcheck_agents/tools/web_search.py` — read-only reference; `include_domains` param already present, no changes needed

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `factcheck_agents/tools/web_search.py` `web_search(query, max_results, include_domains)`: pass `include_domains=["twitter.com", "facebook.com"]` and `max_results=3` — no changes to this module.
- `factcheck_agents/agents/search_agent.py` `_fetch_evidence_image(url)`: extract to shared helper (D-05); function already silently handles all failure modes.
- `factcheck_agents/graph_utils.py` `EvidenceGraph.add_node()` / `add_edge()`: incremental mutation API ready — social_search calls these directly rather than `build_from_evidence()` (which rebuilds from scratch).

### Established Patterns
- `seen: set` for URL dedup (first-occurrence wins) — from `search_agent.py`; apply to social results against existing graph node IDs.
- `lru_cache(maxsize=1)` singletons — not needed here (no models); social_search is purely I/O.
- Graceful degrade: any failure (search provider down, image fetch fail) → continue with partial/empty results, never raise.
- `_tier_priority = {"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}` — established in search_agent and phobert_checker; not needed in social_search itself but downstream agents respect it.

### Integration Points
- **Reads from state:** `state["search_queries"]` (List[str]), `state["evidence_graph"]` (EvidenceGraph)
- **Writes to state:** `{"evidence_graph": updated_graph, "messages": [("assistant", "[SocialSearch] ...")]}`
- **Phase 6 `graph.py`** will add: `add_conditional_edges("verify", route_after_verify, {"social_search": "social_search", "conclusion": "conclusion"})` + `add_edge("social_search", "conclusion")` — no changes in Phase 4.
- **Phase 5 `conclusion_agent`** will read `evidence_graph` — social `"mentions"` edges are neutral and won't trigger conflict detection (which reads `"contradicts"` edges).

</code_context>

<specifics>
## Specific Ideas

- Social domain filter: `include_domains=["twitter.com", "facebook.com"]` — matches SOCIAL-01's `site:twitter.com OR site:facebook.com` restriction.
- Max results hard cap: `max_results=3` per `web_search()` call (SOCIAL-01 explicit).
- Evidence item structure: `Evidence(title=..., url=..., snippet=..., source="tavily"|"google_cse", score=..., source_tier="social", image_path=..., image_caption=...)` — same TypedDict as search_agent produces.
- `EvidenceGraph` node ID for social items: use URL (same convention as `build_from_evidence()` in Phase 1).
- Log message pattern: `[SocialSearch] {len(new_items)} new social items merged into graph ({len(queries)} queries x 3 results max)`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 4-Social Search Sub-Node*
*Context gathered: 2026-07-19*
