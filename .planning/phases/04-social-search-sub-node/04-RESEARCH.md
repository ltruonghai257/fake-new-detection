# Phase 4: Social Search Sub-Node — Research

**Phase:** 04 — Social Search Sub-Node
**Requirements:** SOCIAL-01, SOCIAL-02, SOCIAL-03
**Written:** 2026-07-19

---

## RESEARCH COMPLETE

---

## Codebase Findings

### What exists and is ready to reuse

**`factcheck_agents/tools/web_search.py`**
- `web_search(query, max_results=None, include_domains=None)` — already accepts `include_domains: list | None`
- Tavily path: passes `include_domains` directly in payload
- Google CSE path: appends `site:` tokens to query string
- **No changes needed** — just call `web_search(q, max_results=3, include_domains=["twitter.com", "facebook.com"])`

**`factcheck_agents/graph_utils.py` — `EvidenceGraph`**
- `add_node(id, attrs)` — incremental; adds without rebuilding
- `add_edge(src, dst, type, attrs=None)` — incremental
- `build_from_evidence()` — only for fresh builds; social search MUST use incremental API to preserve existing nodes
- Node IDs are URLs (established convention)

**`factcheck_agents/state.py`**
- `FactCheckState` already has `search_queries: List[str]` and `evidence_graph: Optional[Any]`
- `Evidence` TypedDict has `source_tier: Literal["trusted", "flagged", "social", "unknown"]` — "social" is valid
- `image_path: Optional[str]` and `image_caption: Optional[str]` fields exist

**`factcheck_agents/agents/search_agent.py`**
- `_fetch_evidence_image(url)` at lines 45–158 — must be extracted (D-05)
- Function is self-contained (no dependencies on search_agent module scope except `settings` and stdlib)
- Already handles all failure modes via `try/except: return None, None`
- URL dedup pattern: `seen: set` initialized before the loop, `url in seen` guard

**`factcheck_agents/agents/__init__.py`** (current):
```python
from .search_agent import search_agent
from .evaluate_agent import evaluate_agent
from .verify_agent import verify_agent
from .conclusion_agent import conclusion_agent
__all__ = ["search_agent", "evaluate_agent", "verify_agent", "conclusion_agent"]
```
— `social_search_agent` must be added here after creation.

**`factcheck_agents/source_tier.py` — `classify_domain()`**
- Only checks trusted/flagged lists from env config
- Will NOT return "social" for twitter.com/facebook.com (those aren't in the trusted/flagged lists)
- **Do NOT call `classify_domain()` for social items** — assign `source_tier="social"` directly

**`factcheck_agents/graph.py`**
- Currently: `search -> evaluate -> conclusion`
- Phase 4 does NOT touch `graph.py` — that's Phase 6

### What does NOT yet exist
- `factcheck_agents/helpers.py` — doesn't exist yet; must be created for `_fetch_evidence_image`
- `factcheck_agents/agents/social_search_agent.py` — new file

---

## Implementation Approach

### Task sequence

1. **Extract `_fetch_evidence_image`** into `factcheck_agents/helpers.py`
   - Move function verbatim; it only needs `settings`, `requests`, stdlib
   - Update `search_agent.py` to `from ..helpers import _fetch_evidence_image`

2. **Create `social_search_agent.py`**
   - Reads `state["search_queries"]` and `state["evidence_graph"]`
   - Initializes `seen` from existing graph node IDs: `set(eg.graph.nodes)`
   - Loops queries: `web_search(q, max_results=3, include_domains=["twitter.com", "facebook.com"])`
   - Tags each new result `source_tier="social"`
   - Calls `_fetch_evidence_image(url)` per item (mirror search_agent)
   - Calls `eg.add_node(url, {...})` + `eg.add_edge("statement", url, type="mentions")`
   - Returns `{"evidence_graph": eg, "messages": [...]}`

3. **Export** from `agents/__init__.py`

4. **Tests** in `tests/factcheck_agents/test_social_search_agent.py`

### Dedup strategy
Initialize `seen` from `set(eg.graph.nodes)` — covers both statement node and all existing evidence URLs. Skip social result if URL already in `seen`. First-occurrence wins.

### Empty graph guard
`state.get("evidence_graph")` may be `None` if social_search is somehow called before search_agent. Guard: if `None`, build a fresh `EvidenceGraph()` so incremental adds don't crash.

### Node structure for social items
```python
eg.add_node(url, {
    "node_type": "evidence",
    "title": title,
    "snippet": snippet,
    "source_tier": "social",
    "image_path": img_path,
    "image_caption": img_caption,
})
eg.add_edge("statement", url, type="mentions")
```

---

## Validation Architecture

### Test coverage needed

| Test | What it validates |
|------|-------------------|
| `test_social_items_tagged_social` | source_tier=="social" on all results |
| `test_merges_into_existing_graph` | existing nodes preserved, new ones added |
| `test_dedup_skips_existing_urls` | URL already in graph → not re-added |
| `test_max_results_3_per_query` | web_search called with max_results=3 |
| `test_include_domains_passed` | include_domains=["twitter.com","facebook.com"] |
| `test_no_results_returns_unchanged_graph` | empty web_search → unchanged graph |
| `test_returns_evidence_graph_and_messages` | state keys present |
| `test_no_flat_evidence_written` | "evidence" key NOT in return dict |

### Mocking strategy
- `patch("factcheck_agents.agents.social_search_agent.web_search")` — control results
- `patch("factcheck_agents.agents.social_search_agent._fetch_evidence_image", return_value=(None, None))` — skip network

---

## Key Decisions Reaffirmed

- **D-01**: Reuse `state["search_queries"]` — no extra LLM call
- **D-02**: Return `{"evidence_graph": ..., "messages": [...]}` only — no flat evidence write
- **D-03**: Edge type `"mentions"` for social nodes (not "supports"/"contradicts")
- **D-04**: Mirror `_fetch_evidence_image` call per item (image stored as metadata only)
- **D-05**: Extract to `helpers.py` — shared by both agents

---

## Risk Notes

- **Rate limits**: Tavily/Google CSE with `include_domains=["twitter.com","facebook.com"]` may return 0 results (social platforms restrict crawling). Graceful empty-result path is mandatory.
- **No new dependencies**: `requests`, `bs4`, `networkx` already present. No new packages needed.
- **`_fetch_evidence_image` refactor safety**: `search_agent.py` import swap is one-line; no behavioral change.
