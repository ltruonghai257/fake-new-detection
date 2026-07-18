# Requirements: factcheck_agents

**Defined:** 2026-07-19
**Core Value:** A user submits a Vietnamese claim and gets back a binary Thật/Giả verdict with Vietnamese rationale and citations, even when model checkpoints are missing.

## v2.0 Requirements

### Evidence Graph Infrastructure

- [ ] **EVGRAPH-01**: System builds a `networkx.DiGraph` evidence graph per check-run, with nodes for the statement and each evidence snippet, and typed edges (`supports`/`contradicts`/`mentions`) carrying a `tier` attribute
- [ ] **EVGRAPH-02**: Evidence graph is stored in `FactCheckState.evidence_graph` and travels unchanged through the pipeline — downstream agents read it without re-fetching raw results
- [ ] **EVGRAPH-03**: `build_evidence_text()` prioritizes `trusted`-tier snippet nodes over `flagged` and `unknown` when assembling the PhoBERT evidence passage

### Source Tier Configuration

- [ ] **CONFIG-01**: Trusted source domains are read from `FACTCHECK_TRUSTED_DOMAINS` env var (comma-separated; default: `vnexpress.net,thanhnien.vn,dantri.com.vn,tuoitre.vn`) following existing env-driven config pattern
- [ ] **CONFIG-02**: Flagged low-reliability domains are read from `FACTCHECK_FLAGGED_DOMAINS` env var (comma-separated; default: `kenh14.vn`)
- [ ] **CONFIG-03**: Each `Evidence` item gains a `source_tier` field (`"trusted"` / `"flagged"` / `"unknown"`) set by the search agent based on the URL's domain

### Search / Evidence Agent

- [ ] **SEARCH-01**: Search agent runs separate queries targeting trusted domains and flagged domains (domain-restricted via `site:` operator or Tavily `include_domains` / `exclude_domains`), tagging each result with its `source_tier`
- [ ] **SEARCH-02**: Search agent drafts queries in Vietnamese when the statement language is Vietnamese (LLM prompt updated to request Vietnamese queries)
- [ ] **SEARCH-03**: After retrieval, search agent builds the evidence graph from tagged results and writes it to `state["evidence_graph"]`

### Verify Agent

- [ ] **VERIFY-01**: Verify agent replaces the evaluate agent; runs PhoBERT ViFactCheck (statement + evidence-graph context text) and COOLANT (image, only when supplied) using `concurrent.futures.ThreadPoolExecutor` so both execute concurrently where checkpoints exist
- [ ] **VERIFY-02**: Verify agent computes `reliability_signal: bool` — `True` when at least one model is available AND its output is not NEI/unavailable AND confidence ≥ configurable threshold (`FACTCHECK_RELIABILITY_THRESHOLD`, default `0.5`)
- [ ] **VERIFY-03**: Verify agent writes `reliability_signal` and `model_results` to state; does not call social search itself (routing is graph-level)

### Social Search Sub-Node

- [ ] **SOCIAL-01**: A `social_search` node runs site-restricted queries (`site:twitter.com OR site:facebook.com`) through the existing Tavily/Google CSE tools (no new API dependency) — max 3 results per query to limit rate usage
- [ ] **SOCIAL-02**: Social search results are tagged `source_tier="social"` and merged into the existing `evidence_graph` (new nodes + edges added to the existing DiGraph in state)
- [ ] **SOCIAL-03**: Social search node is only reached when `reliability_signal` is `True`; it is skipped transparently otherwise (graph routing handles this — no flag-check inside the node)

### Conclusion Agent

- [ ] **CONCL-01**: Conclusion agent reads the evidence graph (via `state["evidence_graph"]`) and inspects edge types to detect cross-source conflicts: if any `contradicts` edge exists between a trusted-tier source and the statement → Giả
- [ ] **CONCL-02**: Conclusion agent applies the binary rule: all checked sources/signals consistent with statement → `"REAL"` / `"Thật"`; any conflict → `"FAKE"` / `"Giả"`
- [ ] **CONCL-03**: `Verdict` dict gains two new fields: `verdict_binary: "REAL" | "FAKE"` and `verdict_label_vi: "Thật" | "Giả"`; existing `label` field (4-class) is preserved for backward compatibility
- [ ] **CONCL-04**: LLM conclusion prompt instructs the model to write `rationale` and `recommendation` in Vietnamese; the prompt includes the 4-class internal signal as rationale context but the primary output is the binary verdict
- [ ] **CONCL-05**: Rule-based fallback (no LLM) also applies the binary mapping: REFUTED/FAKE → `"FAKE"`, SUPPORTED/REAL/NEI/UNVERIFIED → follows the "not verifiably real = FAKE" default for UNVERIFIED
- [ ] **CONCL-06**: `_fallback_verdict` is updated to return `verdict_binary` and `verdict_label_vi` alongside the existing `label`

### LangGraph Wiring

- [ ] **GRAPH-01**: `graph.py` replaces `evaluate_agent` node with `verify_agent` node and adds a `social_search` node
- [ ] **GRAPH-02**: `add_conditional_edges("verify", route_after_verify, {"social_search": "social_search", "conclusion": "conclusion"})` — `route_after_verify(state)` reads `state["reliability_signal"]` synchronously and returns the appropriate key
- [ ] **GRAPH-03**: `social_search` node has a regular `add_edge("social_search", "conclusion")`

### Output Surface

- [ ] **OUTPUT-01**: CLI `_print_human()` displays `verdict_label_vi` ("Thật" / "Giả") as the primary label; shows 4-class `label` as a parenthetical detail
- [ ] **OUTPUT-02**: JSON output (`--json` flag) includes `verdict_binary` and `verdict_label_vi` fields alongside existing fields — no fields removed
- [ ] **OUTPUT-03**: `run_fact_check()` Python API return dict includes `verdict_binary` and `verdict_label_vi` at the top level (alongside `verdict` dict) for easy access without breaking existing callers
- [ ] **OUTPUT-04**: MCP `fact_check` tool response includes `verdict_binary` and `verdict_label_vi` alongside existing `verdict` dict

### Tests

- [ ] **TEST-01**: Unit tests cover source-tier tagging logic (`trusted`/`flagged`/`unknown` from URL domain)
- [ ] **TEST-02**: Unit tests cover evidence graph construction (nodes, edge types, tier attributes)
- [ ] **TEST-03**: Unit tests cover `reliability_signal` computation (model available + confident → True; all unavailable → False; NEI only → False)
- [ ] **TEST-04**: Unit tests cover binary verdict mapping (SUPPORTED→REAL, REFUTED→FAKE, NEI→FAKE, UNVERIFIED→FAKE, MISLEADING→FAKE)
- [ ] **TEST-05**: Unit tests verify `verdict_label_vi` is `"Thật"` when verdict is REAL and `"Giả"` when FAKE
- [ ] **TEST-06**: Integration test verifies graceful degrade: with no checkpoints, pipeline still returns `verdict_binary` and `verdict_label_vi` (not crash, not missing fields)

## v3+ Requirements (deferred)

### Social

- **SOC-01**: X/Twitter or Meta Graph API integration (paid) — explicit confirmation required before adding
- **SOC-02**: Real-time streaming social media monitoring

### Multilingual

- **LANG-01**: Auto-detect statement language and generate rationale in matching language (beyond Vietnamese)

## Out of Scope

| Feature | Reason |
|---------|--------|
| `@colbymchenry/codegraph` npm package | Indexes source code (ASTs), not text evidence — plain Python graph only |
| Paid X/Twitter API or Meta Graph API | Stop condition — requires explicit confirmation |
| Neo4j or other external graph DB | Overkill for per-request lifecycle; NetworkX in-memory is sufficient |
| Modifying `training/`, model checkpoints, or notebooks | Strict decoupling — factcheck_agents must not depend on training pipeline |
| Changing existing CLI/API/MCP response shape destructively | All new fields are additive; existing callers must not break |
| Parallel LangGraph async nodes | TradingAgents confirms sequential nodes + state flags is the correct pattern |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVGRAPH-01 | Phase 1 | Pending |
| EVGRAPH-02 | Phase 1 | Pending |
| EVGRAPH-03 | Phase 1 | Pending |
| CONFIG-01 | Phase 1 | Pending |
| CONFIG-02 | Phase 1 | Pending |
| CONFIG-03 | Phase 1 | Pending |
| SEARCH-01 | Phase 2 | Pending |
| SEARCH-02 | Phase 2 | Pending |
| SEARCH-03 | Phase 2 | Pending |
| VERIFY-01 | Phase 3 | Pending |
| VERIFY-02 | Phase 3 | Pending |
| VERIFY-03 | Phase 3 | Pending |
| SOCIAL-01 | Phase 4 | Pending |
| SOCIAL-02 | Phase 4 | Pending |
| SOCIAL-03 | Phase 4 | Pending |
| CONCL-01 | Phase 5 | Pending |
| CONCL-02 | Phase 5 | Pending |
| CONCL-03 | Phase 5 | Pending |
| CONCL-04 | Phase 5 | Pending |
| CONCL-05 | Phase 5 | Pending |
| CONCL-06 | Phase 5 | Pending |
| GRAPH-01 | Phase 6 | Pending |
| GRAPH-02 | Phase 6 | Pending |
| GRAPH-03 | Phase 6 | Pending |
| OUTPUT-01 | Phase 7 | Pending |
| OUTPUT-02 | Phase 7 | Pending |
| OUTPUT-03 | Phase 7 | Pending |
| OUTPUT-04 | Phase 7 | Pending |
| TEST-01 | Phase 8 | Pending |
| TEST-02 | Phase 8 | Pending |
| TEST-03 | Phase 8 | Pending |
| TEST-04 | Phase 8 | Pending |
| TEST-05 | Phase 8 | Pending |
| TEST-06 | Phase 8 | Pending |

**Coverage:**
- v2.0 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-19*
*Last updated: 2026-07-19 after initial definition*
