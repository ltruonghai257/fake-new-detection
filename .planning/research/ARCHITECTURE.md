# Architecture Research — v2.0 Evidence-Graph Pipeline

## TradingAgents Pattern (deepwiki confirmed)

TradingAgents uses **sequential nodes** sharing a single `AgentState` TypedDict — not true parallelism at the LangGraph level. Each analyst node writes its own slice of state, and a `ConditionalLogic` class routes edges based on state flags.

Key patterns to mirror:
- One `TypedDict` threaded through all nodes — each node writes its own slice
- Conditional edges: `add_conditional_edges(source, path_fn, {key: node_name})` — `path_fn` reads state and returns a single string key
- "Clear node" between analyst stages to reset message accumulation
- Debate/research team pattern: node sets `reliability_signal` flag → router reads it → branches to social-search sub-node or skips

## LangGraph Conditional Edge Pattern (confirmed, langgraph 0.2.x)

```python
def route_after_verify(state: FactCheckState) -> str:
    if state.get("reliability_signal"):
        return "social_search"
    return "conclusion"

g.add_conditional_edges("verify", route_after_verify, {
    "social_search": "social_search",
    "conclusion": "conclusion",
})
g.add_edge("social_search", "conclusion")
```

**Pitfalls:**
- Path function must return EXACTLY one key matching the path_map; unmapped keys raise `KeyError` at invoke-time (not compile-time)
- Do NOT make path function async — move async logic into preceding node, set flag in state, route synchronously on flag
- Returning `END` constant is valid; returning a list was removed in 0.2.x

## Evidence Graph Pattern

Research shows **NetworkX DiGraph** is the standard for in-memory evidence graphs (confirmed: EVOCA paper, ClaimVer, BoggersTheCIG all use NetworkX). For our lightweight per-request use case:

```python
import networkx as nx

G = nx.DiGraph()
# nodes: entities (statement, source_domains, evidence_snippets)
# edges: ("statement", snippet_id, {"relation": "mentions"/"supports"/"contradicts", "tier": "trusted"})
```

**Alternatives considered:**
- Plain `dict` of dicts — sufficient if we don't need graph traversal algorithms
- SQLite — overkill for a single request lifecycle
- Neo4j — far too heavy, requires external process

**Decision**: Use `networkx` for the evidence graph; it's already in the Python ecosystem and ClaimVer/EVOCA confirm it's appropriate for this scale. Add as a new dependency (pure Python, no heavy extras).

## ViFactCheck Input Format (from existing phobert_checker.py)

```python
tokenizer(
    statement,           # text_a
    evidence_text,       # text_b — concatenated snippets, up to 2000 chars
    truncation="only_second",   # truncate evidence, not statement
    max_length=256,
)
```

Labels: `{0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}`

For evidence-graph context, `build_evidence_text()` already concatenates snippets. In v2.0 it should prefer `trusted`-tier snippets first, then `flagged`, then `unknown` when building the evidence passage.

## Integration Points for v2.0

| Existing | Change |
|----------|--------|
| `FactCheckState.evidence: List[Evidence]` | Add `evidence_graph: Optional[Any]` (NetworkX DiGraph stored in state) |
| `search_agent` → query + fetch | Split into: query-per-tier, tag hits, build graph |
| `evaluate_agent` → sequential models | Rename to `verify_agent`, add `reliability_signal: bool` to state |
| `conclusion_agent` → 4-class verdict | Add binary mapping + `verdict_label_vi` + Vietnamese prompts |
| `graph.py` → linear edges | Add `add_conditional_edges("verify", route_fn, ...)` |

## Build Order

1. State + Config changes (evidence_graph field, source-tier env vars)
2. Evidence graph structure (EvidenceGraph class or plain nx.DiGraph wrapper)
3. Search/Evidence Agent (tier queries, graph build)
4. Verify Agent (parallel model run, reliability_signal)
5. Social search sub-node (site-restricted, conditional)
6. Conclusion Agent (binary verdict, Vietnamese prompts)
7. Graph wiring (conditional edge)
8. Output layer (CLI/API/MCP additive changes)
9. Tests
