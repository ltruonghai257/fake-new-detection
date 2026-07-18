# Roadmap — factcheck_agents v2.0 Evidence-Graph Vietnamese Pipeline

**Milestone:** v2.0 · **Status:** Active

---

## Phase 1 — State, Config & Evidence Graph Foundation

**Goal:** Extend `FactCheckState` and `config.py` with new fields; implement the in-memory evidence graph structure.

**Requirements covered:** EVGRAPH-01, EVGRAPH-02, EVGRAPH-03, CONFIG-01, CONFIG-02, CONFIG-03

**Deliverables:**
- `state.py`: add `evidence_graph: Optional[Any]`, `reliability_signal: Optional[bool]`, extend `Evidence` with `source_tier`
- `config.py`: add `FACTCHECK_TRUSTED_DOMAINS`, `FACTCHECK_FLAGGED_DOMAINS`, `FACTCHECK_RELIABILITY_THRESHOLD` env vars
- `factcheck_agents/graph_utils.py` (new): `EvidenceGraph` wrapper around `networkx.DiGraph` — build, add node, add edge, to_evidence_list
- `factcheck_agents/source_tier.py` (new): `classify_domain(url) -> "trusted"|"flagged"|"social"|"unknown"` from config

**Verify:** Unit tests pass — tier tagging for sample URLs, evidence graph node/edge creation (TEST-01, TEST-02)

---

## Phase 2 — Search / Evidence Agent

**Goal:** Rewrite the search agent to run tier-separated queries, tag results, build the evidence graph, and store it in state.

**Requirements covered:** SEARCH-01, SEARCH-02, SEARCH-03

**Deliverables:**
- `agents/search_agent.py`: separate trusted-domain and flagged-domain queries; tag each `Evidence` item with `source_tier`; call `EvidenceGraph.build_from_evidence()` and write to `state["evidence_graph"]`
- `prompts.py`: update `SEARCH_QUERY_PROMPT` to instruct Vietnamese queries
- `tools/web_search.py`: add `include_domains` / `exclude_domains` passthrough for Tavily and Google CSE

**Verify:** Search agent produces `state["evidence_graph"]` with tagged nodes on a sample statement (no model checkpoint needed)

---

## Phase 3 — Verify Agent

**Goal:** Replace `evaluate_agent` with `verify_agent` — concurrent model execution, `reliability_signal` computation.

**Requirements covered:** VERIFY-01, VERIFY-02, VERIFY-03

**Deliverables:**
- `agents/verify_agent.py` (new): `ThreadPoolExecutor` runs PhoBERT + COOLANT concurrently; `reliability_signal` computed from fused outputs; writes both to state
- `agents/evaluate_agent.py`: kept in place but `graph.py` will wire `verify_agent` instead
- `models/phobert_checker.py`: update `build_evidence_text()` to prefer trusted-tier snippets first (EVGRAPH-03)

**Verify:** With no checkpoints → `reliability_signal=False`, pipeline does not crash (TEST-03, TEST-06 partial)

---

## Phase 4 — Social Search Sub-Node

**Goal:** Implement the `social_search` node — site-restricted Tavily/Google CSE queries, tier-tagged results merged into the evidence graph.

**Requirements covered:** SOCIAL-01, SOCIAL-02, SOCIAL-03

**Deliverables:**
- `agents/social_search_agent.py` (new): queries `site:twitter.com OR site:facebook.com`, max 3 results per query; tags `source_tier="social"`; merges new nodes into `state["evidence_graph"]`

**Verify:** `social_search` node returns state with merged social nodes; never called when `reliability_signal=False`

---

## Phase 5 — Conclusion Agent (Binary Verdict + Vietnamese)

**Goal:** Extend the conclusion agent to produce a binary verdict, `verdict_label_vi`, and Vietnamese rationale.

**Requirements covered:** CONCL-01, CONCL-02, CONCL-03, CONCL-04, CONCL-05, CONCL-06

**Deliverables:**
- `agents/conclusion_agent.py`: read `state["evidence_graph"]` for cross-source conflict detection; apply binary rule; write `verdict_binary` and `verdict_label_vi` to `Verdict`
- `state.py`: add `verdict_binary` and `verdict_label_vi` fields to `Verdict` TypedDict
- `prompts.py`: update `CONCLUSION_SYSTEM_PROMPT` to request Vietnamese rationale and include binary verdict instruction
- `agents/conclusion_agent.py`: update `_fallback_verdict` to also return binary fields

**Verify:** Unit tests for binary mapping (SUPPORTED→REAL, REFUTED→FAKE, NEI/UNVERIFIED/MISLEADING→FAKE) and Vietnamese label presence (TEST-04, TEST-05)

---

## Phase 6 — LangGraph Wiring

**Goal:** Rewire `graph.py` with the new node names and conditional edge for social search.

**Requirements covered:** GRAPH-01, GRAPH-02, GRAPH-03

**Deliverables:**
- `graph.py`: replace `evaluate` node with `verify`; add `social_search` node; add `add_conditional_edges("verify", route_after_verify, {"social_search": ..., "conclusion": ...})`; add `add_edge("social_search", "conclusion")`
- `graph.py`: `route_after_verify(state)` reads `state["reliability_signal"]` synchronously

**Verify:** Graph compiles without error; invoke with no checkpoints routes directly to conclusion (not social_search)

---

## Phase 7 — Output Surface

**Goal:** Surface `verdict_binary` and `verdict_label_vi` in CLI, Python API, and MCP server additively.

**Requirements covered:** OUTPUT-01, OUTPUT-02, OUTPUT-03, OUTPUT-04

**Deliverables:**
- `cli.py`: `_print_human()` shows `verdict_label_vi` ("Thật"/"Giả") as primary, 4-class label as parenthetical; `--json` output includes new fields
- `__init__.py` / `run_fact_check()`: return dict includes `verdict_binary` and `verdict_label_vi` at top level
- `mcp_server.py`: `fact_check` tool response includes new fields
- `factcheck_agents/README.md`: update example outputs to show new fields

**Verify:** CLI `--json` output includes both new fields; existing `verdict` dict still present; no callers broken

---

## Phase 8 — Tests

**Goal:** Write unit and integration tests covering all new behaviour.

**Requirements covered:** TEST-01 through TEST-06

**Deliverables:**
- `tests/test_source_tier.py`: tier classification from URL domains
- `tests/test_evidence_graph.py`: graph construction, node/edge attributes
- `tests/test_reliability_signal.py`: signal computation under all model availability combinations
- `tests/test_binary_verdict.py`: 4-class → binary mapping + Vietnamese labels
- `tests/test_graceful_degrade.py`: full pipeline run with no checkpoints → verdict present, no crash

**Verify:** `pytest tests/` passes; all TEST-01..06 requirements satisfied

---

## Dependency Map

```
Phase 1 (State+Config+Graph) ──► Phase 2 (Search)
                               ──► Phase 3 (Verify)
Phase 2 ──► Phase 4 (Social)
Phase 3 ──► Phase 4 (Social)
Phase 3 ──► Phase 5 (Conclusion)
Phase 4 ──► Phase 5 (Conclusion)
Phase 5 ──► Phase 6 (Wiring)
Phase 6 ──► Phase 7 (Output)
Phase 6 ──► Phase 8 (Tests)
Phase 7 ──► Phase 8 (Tests)
```

Phases 2 and 3 can be worked in parallel after Phase 1.  
Phase 4 requires both 2 and 3.

---

*Created: 2026-07-19*
