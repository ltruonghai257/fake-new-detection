# Plan 06-01 Summary: Rewire graph.py with verify node and social_search conditional routing

## Objective
Rewire factcheck_agents/graph.py to replace the `evaluate` node with `verify`, add the `social_search` node, and wire the conditional edge from `verify` → `social_search`/`conclusion` via `route_after_verify`.

## Changes Made

### factcheck_agents/graph.py
1. Updated module-level docstring from "Search -> Evaluate -> Conclusion" to "Search -> Verify → (Social?) → Conclusion"
2. Updated imports:
   - Removed `evaluate_agent`
   - Added `verify_agent` and `social_search_agent`
3. Added `route_after_verify(state: FactCheckState) -> str` function that:
   - Returns `"social_search"` when `state["reliability_signal"]` is truthy
   - Returns `"conclusion"` when `state["reliability_signal"]` is falsy or None
4. Replaced `evaluate` node with `verify` node in `build_graph()`
5. Added `social_search` node in `build_graph()`
6. Replaced edges:
   - `search → evaluate` → `search → verify`
   - `evaluate → conclusion` → conditional routing via `route_after_verify`
   - Added `verify → social_search` (conditional) and `verify → conclusion` (conditional)
   - Added `social_search → conclusion`
7. Kept `conclusion → END` and `return g.compile()` unchanged

## Verification Results
All verification commands passed:
- `build_graph()` compiles successfully
- `route_after_verify` routes correctly:
  - `reliability_signal=True` → `"social_search"`
  - `reliability_signal=False` → `"conclusion"`
  - `reliability_signal=None` → `"conclusion"`
- `evaluate_agent` is no longer imported in graph.py

## Requirements Satisfied
- GRAPH-01: graph.py replaces the `evaluate` node with `verify`
- GRAPH-02: `social_search` node is present with a conditional edge from `verify`
- GRAPH-03: `route_after_verify(state)` returns `social_search` when `reliability_signal=True`, `conclusion` otherwise
- Additional: `add_edge('social_search', 'conclusion')` exists
- Additional: Graph compiles without error after rewiring

## Artifacts
- `factcheck_agents/graph.py` - Rewired LangGraph with verify + social_search nodes and conditional routing

## Commit
Commit hash: 1665dff

## Status
✅ Complete
