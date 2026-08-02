# Phase 4: Social Search Sub-Node - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-19
**Phase:** 4-Social Search Sub-Node
**Areas discussed:** Query source, State write-back, Social node edge type, Image fetching

---

## Query Source

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse `state["search_queries"]` | Already drafted in Vietnamese by search_agent's LLM call — no extra cost, multiple focused queries | ✓ |
| Use `state["statement"]` directly | Single query, no dependency on search_queries, simpler but less focused | |
| Draft new social-specific queries (LLM call) | Tailored for social media, but extra cost and complexity | |

**User's choice:** Reuse `state["search_queries"]`
**Notes:** Avoids extra LLM call; heuristic fallback in search_agent guarantees search_queries is always non-empty.

---

## State Write-Back

| Option | Description | Selected |
|--------|-------------|----------|
| Graph only | Don't touch `state["evidence"]`; Phase 5 reads evidence_graph directly | ✓ |
| Append to flat list too | Mirror search_agent's both-write pattern for backward compat | |

**User's choice:** Graph only
**Notes:** Consistent with SOCIAL-02 ("merged into the existing evidence_graph"). Phase 5 conclusion_agent is designed to read the graph.

---

## Social Node Edge Type

| Option | Description | Selected |
|--------|-------------|----------|
| `"mentions"` | Safe default — social posts unverified by any model; neutral for Phase 5 conflict detection | ✓ |
| Leave to Claude | Planner/executor decides; risk of Phase 5 conflict detection misfiring | |

**User's choice:** `"mentions"`
**Notes:** Phase 5 reads `"contradicts"` edges for conflict detection; `"mentions"` edges are safely neutral.

---

## Image Fetching for Social Evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Skip image fetching | COOLANT already ran; Twitter/Facebook URLs are scrape-resistant; simpler | |
| Fetch images + captions (mirror search_agent) | Consistent with search_agent pattern; stores metadata even if COOLANT can't use it now | ✓ |

**User's choice:** Fetch images + captions, mirror search_agent
**Notes:** COOLANT runs in Phase 3 (before social_search), so images are graph metadata only for this phase. `_fetch_evidence_image()` to be extracted from search_agent.py into a shared helper module (D-05).

---

## Claude's Discretion

- **Dedup strategy:** `seen: set` of existing evidence_graph node URLs before looping queries; first-occurrence wins — same pattern as search_agent.
- **No-results behavior:** return unchanged `evidence_graph` with diagnostic message; never raise.
- **`_fetch_evidence_image()` location:** Extract to `factcheck_agents/helpers.py` (or equivalent shared module); update search_agent.py import.

## Deferred Ideas

None — discussion stayed within phase scope.
