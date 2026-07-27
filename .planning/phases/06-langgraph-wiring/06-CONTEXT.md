# Phase 6: LangGraph Wiring - Context

**Gathered:** 2026-07-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Rewire `factcheck_agents/graph.py` to replace the `evaluate_agent` node with `verify_agent`, add the `social_search` node, wire the conditional edge from `verify` → `social_search`/`conclusion`, and add `SqliteSaver`/`MemorySaver` checkpointer support so interrupted runs resume from the last completed node. The phase scope is **`graph.py` only** — no changes to agent implementations, callers, CLI, or MCP server.

</domain>

<decisions>
## Implementation Decisions

### Checkpointer DB Path
- **D-01:** `FACTCHECK_CHECKPOINT_DB` env var controls the SqliteSaver file path; default: `.factcheck_checkpoints.db` in the project root. Follows the existing env-driven config pattern (`FACTCHECK_TRUSTED_DOMAINS`, etc.).
- **D-02:** SqliteSaver import failure (`ImportError` — missing `langgraph[sqlite]` extra) silently falls back to `MemorySaver`. No warning log. Consistent with the graceful-degrade pattern throughout the codebase.

### build_graph() Signature
- **D-03:** `build_graph(checkpointer=None)` — when `None`, creates the default checkpointer (SqliteSaver/MemorySaver) internally. Callers can pass an explicit checkpointer (e.g., `MemorySaver()` in tests). No breaking change to existing call sites.
- **D-04:** `initial_state()` is unchanged. `thread_id` lives in the LangGraph invoke `config` dict (`config={"configurable": {"thread_id": "..."}}`), not in `FactCheckState`. Phase 7 handles wiring `thread_id` into callers.

### evaluate_agent Cleanup Scope
- **D-05:** Phase 6 removes `evaluate_agent` from `graph.py`'s import only. `factcheck_agents/agents/__init__.py` keeps its existing export — the module is still valid code, just no longer wired into the graph.
- **D-06:** `graph.py` module-level docstring is updated from `"Search -> Evaluate -> Conclusion"` to `"Search -> Verify -> (Social?) -> Conclusion"`.

### thread_id Wiring Boundary
- **D-07:** Phase 6 stops at `build_graph()`. No changes to `cli.py`, `factcheck_agents/__init__.py`, or `mcp_server.py`. Thread_id wiring into invoke calls is Phase 7 scope.
- **D-08:** Phase 6 executor verifies that calling `g.invoke(state)` without a `config` dict does not crash when a checkpointer is compiled in (LangGraph uses a null default thread). Findings documented as a `# Note for Phase 7:` comment in `graph.py`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §LangGraph Wiring (GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04) — exact node names, conditional edge signature, checkpointer fallback contract

### Prior Phase Context
- `.planning/phases/04-social-search-sub-node/04-CONTEXT.md` — D-code context confirming Phase 6 is responsible for `add_conditional_edges("verify", route_after_verify, {...})` + `add_edge("social_search", "conclusion")`
- `.planning/phases/03-verify-agent/03-CONTEXT.md` — `reliability_signal` field name and type (`bool`) that `route_after_verify` reads from state
- `.planning/phases/05-conclusion-agent/05-CONTEXT.md` — `Verdict` shape and `verdict_binary`/`verdict_label_vi` fields (downstream node)

### Existing Code to Modify
- `factcheck_agents/graph.py` — only file in scope for Phase 6
- `factcheck_agents/config.py` — add `FACTCHECK_CHECKPOINT_DB` env var read with default
- `factcheck_agents/state.py` — read-only reference for `FactCheckState` and `reliability_signal` field

</canonical_refs>

<code_context>
## Existing Code Insights

### Current graph.py Shape (42 lines)
```python
g = StateGraph(FactCheckState)
g.add_node("search", search_agent)
g.add_node("evaluate", evaluate_agent)   # ← replaced by verify_agent
g.add_node("conclusion", conclusion_agent)
g.add_edge(START, "search")
g.add_edge("search", "evaluate")          # ← becomes "search" → "verify"
g.add_edge("evaluate", "conclusion")      # ← replaced by conditional edge
g.add_edge("conclusion", END)
return g.compile()                        # ← gains checkpointer=...
```

### New Node Topology
```
START → search → verify → route_after_verify → social_search → conclusion → END
                                            ↘ conclusion (direct, when reliability_signal=False)
```

### Reusable Assets
- `factcheck_agents/agents/__init__.py`: already exports `verify_agent`, `social_search_agent`, `conclusion_agent` — just update the import in `graph.py`
- `factcheck_agents/config.py`: existing pattern for env var reads with defaults (see `FACTCHECK_TRUSTED_DOMAINS`) — add `FACTCHECK_CHECKPOINT_DB` the same way

### Established Patterns
- All config from env vars with sensible defaults — `FACTCHECK_CHECKPOINT_DB` follows this
- `lru_cache(maxsize=1)` for singletons — the compiled graph object in `__init__.py`'s `run_fact_check` already caches it; `build_graph()` signature change is backward compatible
- Graceful degrade: silent `except ImportError → MemorySaver`; never raise on missing extras

### Integration Points
- **Callers of `build_graph()`:** `factcheck_agents/__init__.py` (`run_fact_check`), `factcheck_agents/cli.py`, `factcheck_agents/mcp_server.py` — all call `build_graph()` with no args; signature `build_graph(checkpointer=None)` is backward compatible
- **`route_after_verify(state: FactCheckState) → str`:** reads `state["reliability_signal"]`; returns `"social_search"` if `True`, `"conclusion"` if `False`/`None`

</code_context>

<specifics>
## Specific Ideas

- `route_after_verify` key mapping: `{"social_search": "social_search", "conclusion": "conclusion"}` per REQUIREMENTS GRAPH-02
- `FACTCHECK_CHECKPOINT_DB` default: `".factcheck_checkpoints.db"` (project root; should be gitignored)
- SqliteSaver import guard:
  ```python
  try:
      from langgraph.checkpoint.sqlite import SqliteSaver
      checkpointer = SqliteSaver.from_conn_string(db_path)
  except ImportError:
      from langgraph.checkpoint.memory import MemorySaver
      checkpointer = MemorySaver()
  ```

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 6-LangGraph Wiring*
*Context gathered: 2026-07-27*
