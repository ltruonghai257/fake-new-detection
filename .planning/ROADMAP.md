# Roadmap: factcheck_agents v2.0 Evidence-Graph Vietnamese Pipeline

**Milestone:** v2.0 · **Status:** Active

## Overview

Extend the `factcheck_agents` pipeline with an evidence graph, source-tier classification, concurrent model verification, social search, binary verdicts with Vietnamese labels, and full test coverage.

## Phases

-   [x] **Phase 1: State, Config & Evidence Graph Foundation** - Extend FactCheckState/config; implement in-memory evidence graph
-   [ ] **Phase 2: Search / Evidence Agent** - Tier-separated queries, tagged results, evidence graph in state _(Planned)_
-   [ ] **Phase 3: Verify Agent** - Concurrent PhoBERT+COOLANT execution, reliability_signal computation
-   [ ] **Phase 4: Social Search Sub-Node** - Site-restricted social queries merged into evidence graph
-   [ ] **Phase 5: Conclusion Agent (Binary Verdict + Vietnamese)** - Binary verdict, verdict_label_vi, Vietnamese rationale
-   [ ] **Phase 6: LangGraph Wiring** - Rewire graph.py with new nodes and conditional edge
-   [ ] **Phase 7: Output Surface** - Surface new fields in CLI, Python API, MCP server
-   [ ] **Phase 8: Tests** - Unit and integration tests for all new behaviour

## Phase Details

### Phase 1: State, Config & Evidence Graph Foundation

**Goal**: Extend `FactCheckState` and `config.py` with new fields; implement the in-memory evidence graph structure.
**Requirements**: EVGRAPH-01, EVGRAPH-02, EVGRAPH-03, CONFIG-01, CONFIG-02, CONFIG-03
**Depends on**: Nothing (first phase)
**Success Criteria** (what must be TRUE):

1. `state.py` adds `evidence_graph: Optional[Any]`, `reliability_signal: Optional[bool]`, and `source_tier` to `Evidence`
2. `config.py` reads `FACTCHECK_TRUSTED_DOMAINS`, `FACTCHECK_FLAGGED_DOMAINS`, `FACTCHECK_RELIABILITY_THRESHOLD` from env
3. `graph_utils.py` `EvidenceGraph` supports build, add_node, add_edge, to_evidence_list
4. `source_tier.py` `classify_domain(url)` returns correct tier for sample URLs
5. Unit tests pass for tier tagging and evidence graph node/edge creation (TEST-01, TEST-02)
   **Plans**: TBD

Plans:

-   [x] 01-01: state.py and config.py field extensions
-   [x] 01-02: graph_utils.py and source_tier.py new modules

---

### Phase 2: Search / Evidence Agent

**Goal**: Rewrite the search agent to run tier-separated queries, tag results, build the evidence graph, and store it in state.
**Requirements**: SEARCH-01, SEARCH-02, SEARCH-03
**Depends on**: Phase 1
**Success Criteria** (what must be TRUE):

1. `search_agent.py` runs separate trusted-domain and flagged-domain queries
2. Each `Evidence` item tagged with `source_tier`
3. `EvidenceGraph.build_from_evidence()` called and result written to `state["evidence_graph"]`
4. `SEARCH_QUERY_PROMPT` instructs Vietnamese queries
5. `web_search.py` passes `include_domains` / `exclude_domains` to Tavily and Google CSE
6. Search agent produces `state["evidence_graph"]` with tagged nodes on a sample statement
   **Plans**: 02-01, 02-02

Plans:

-   [ ] 02-01: search_agent.py tier-separated queries and evidence graph integration
-   [ ] 02-02: prompts.py and web_search.py domain filter updates

---

### Phase 3: Verify Agent

**Goal**: Replace `evaluate_agent` with `verify_agent` — concurrent model execution, `reliability_signal` computation.
**Requirements**: VERIFY-01, VERIFY-02, VERIFY-03
**Depends on**: Phase 1
**Success Criteria** (what must be TRUE):

1. `verify_agent.py` runs PhoBERT + COOLANT concurrently via `ThreadPoolExecutor`
2. `reliability_signal` computed from fused outputs and written to state
3. `phobert_checker.py` `build_evidence_text()` prefers trusted-tier snippets first
4. With no checkpoints → `reliability_signal=False`, pipeline does not crash (TEST-03, TEST-06 partial)
   **Plans**: TBD

Plans:

-   [ ] 03-01: verify_agent.py with concurrent execution and reliability_signal
-   [ ] 03-02: phobert_checker.py trusted-tier evidence ordering

---

### Phase 4: Social Search Sub-Node

**Goal**: Implement the `social_search` node — site-restricted Tavily/Google CSE queries, tier-tagged results merged into the evidence graph.
**Requirements**: SOCIAL-01, SOCIAL-02, SOCIAL-03
**Depends on**: Phase 2, Phase 3
**Success Criteria** (what must be TRUE):

1. `social_search_agent.py` queries `site:twitter.com OR site:facebook.com`, max 3 results per query
2. Results tagged `source_tier="social"` and merged into `state["evidence_graph"]`
3. `social_search` node never called when `reliability_signal=False`
   **Plans**: TBD

Plans:

-   [ ] 04-01: social_search_agent.py implementation

---

### Phase 5: Conclusion Agent (Binary Verdict + Vietnamese)

**Goal**: Extend the conclusion agent to produce a binary verdict, `verdict_label_vi`, and Vietnamese rationale.
**Requirements**: CONCL-01, CONCL-02, CONCL-03, CONCL-04, CONCL-05, CONCL-06
**Depends on**: Phase 3, Phase 4
**Success Criteria** (what must be TRUE):

1. `conclusion_agent.py` reads `state["evidence_graph"]` for cross-source conflict detection
2. Binary rule applied: SUPPORTED→REAL, REFUTED→FAKE, NEI/UNVERIFIED/MISLEADING→FAKE
3. `verdict_binary` and `verdict_label_vi` written to `Verdict`
4. `state.py` `Verdict` TypedDict includes `verdict_binary` and `verdict_label_vi`
5. `CONCLUSION_SYSTEM_PROMPT` requests Vietnamese rationale with binary verdict instruction
6. `_fallback_verdict` returns binary fields (TEST-04, TEST-05)
   **Plans**: TBD

Plans:

-   [ ] 05-01: state.py Verdict fields + conclusion_agent.py binary rule and evidence graph
-   [ ] 05-02: prompts.py CONCLUSION_SYSTEM_PROMPT Vietnamese and binary update

---

### Phase 6: LangGraph Wiring

**Goal**: Rewire `graph.py` with the new node names and conditional edge for social search.
**Requirements**: GRAPH-01, GRAPH-02, GRAPH-03
**Depends on**: Phase 5
**Success Criteria** (what must be TRUE):

1. `graph.py` replaces `evaluate` node with `verify`
2. `social_search` node added with conditional edge from `verify`
3. `route_after_verify(state)` reads `state["reliability_signal"]` synchronously
4. Graph compiles without error
5. Invoke with no checkpoints routes directly to conclusion (not social_search)
   **Plans**: TBD

Plans:

-   [ ] 06-01: graph.py rewiring with new nodes and conditional edges

---

### Phase 7: Output Surface

**Goal**: Surface `verdict_binary` and `verdict_label_vi` in CLI, Python API, and MCP server additively.
**Requirements**: OUTPUT-01, OUTPUT-02, OUTPUT-03, OUTPUT-04
**Depends on**: Phase 6
**Success Criteria** (what must be TRUE):

1. `cli.py` `_print_human()` shows `verdict_label_vi` as primary, 4-class label as parenthetical
2. `--json` output includes `verdict_binary` and `verdict_label_vi`
3. `run_fact_check()` return dict includes both new fields at top level
4. `mcp_server.py` `fact_check` tool response includes new fields
5. Existing `verdict` dict still present; no callers broken
   **Plans**: TBD

Plans:

-   [ ] 07-01: cli.py, **init**.py, mcp_server.py, README.md output updates

---

### Phase 8: Tests

**Goal**: Write unit and integration tests covering all new behaviour.
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06
**Depends on**: Phase 6, Phase 7
**Success Criteria** (what must be TRUE):

1. `tests/test_source_tier.py` passes tier classification from URL domains
2. `tests/test_evidence_graph.py` passes graph construction, node/edge attributes
3. `tests/test_reliability_signal.py` passes signal computation under all model availability combinations
4. `tests/test_binary_verdict.py` passes 4-class → binary mapping + Vietnamese labels
5. `tests/test_graceful_degrade.py` passes full pipeline with no checkpoints
6. `pytest tests/` passes; all TEST-01..06 requirements satisfied
   **Plans**: TBD

Plans:

-   [ ] 08-01: test_source_tier.py and test_evidence_graph.py
-   [ ] 08-02: test_reliability_signal.py, test_binary_verdict.py, test_graceful_degrade.py

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

## Progress

| Phase                                             | Plans Complete | Status      | Completed  |
| ------------------------------------------------- | -------------- | ----------- | ---------- |
| 1. State, Config & Evidence Graph Foundation      | 2/2            | Complete    | 2026-07-19 |
| 2. Search / Evidence Agent                        | 2/2            | Planned     | -          |
| 3. Verify Agent                                   | 0/2            | Not started | -          |
| 4. Social Search Sub-Node                         | 0/1            | Not started | -          |
| 5. Conclusion Agent (Binary Verdict + Vietnamese) | 0/2            | Not started | -          |
| 6. LangGraph Wiring                               | 0/1            | Not started | -          |
| 7. Output Surface                                 | 0/1            | Not started | -          |
| 8. Tests                                          | 0/2            | Not started | -          |

_Created: 2026-07-19_
