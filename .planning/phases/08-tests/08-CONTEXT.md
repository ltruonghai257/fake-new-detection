# Phase 8: Tests - Context

**Gathered:** 2026-07-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Write unit and integration tests covering all new v2.0 behaviour. Most of TEST-01..05 already exist in `tests/factcheck_agents/` from prior phases. Phase 8 fills the gaps: TEST-06 (graceful-degrade integration test), graph wiring unit tests (Phase 6 coverage), and output surface unit tests (Phase 7 coverage). All new test files land in `tests/factcheck_agents/` following the established convention.

No changes to production code — this phase is tests-only (plus `tests/` directory files).

</domain>

<decisions>
## Implementation Decisions

### TEST-06: Graceful-Degrade Integration Test
- **D-01:** `test_graceful_degrade.py` calls `build_graph(checkpointer=MemorySaver()).invoke(state)` — full LangGraph graph execution, not a manual agent chain. This exercises the real Phase 6 wiring (node topology, conditional edge, state threading).
- **D-02:** All four agent functions (`search_agent`, `verify_agent`, `social_search_agent`, `conclusion_agent`) are patched at their import-level namespace. Each mock returns a minimal state dict sufficient for the next node.
- **D-03:** The test asserts: no exception is raised AND `result["verdict"]` is a dict. Value-level correctness is delegated to the agent-level unit tests (test_conclusion_agent.py etc.).
- **D-04:** `MemorySaver` is passed explicitly to `build_graph()` to avoid sqlite dependency in CI. This exercises the `build_graph(checkpointer=...)` signature from Phase 6 decision D-03.

### Graph Wiring Tests (Phase 6 coverage)
- **D-05:** `test_graph_wiring.py` unit-tests `route_after_verify` directly: import it from `factcheck_agents.graph`, call with `{"reliability_signal": True}` → assert `"social_search"`, and `{"reliability_signal": False}` (and `None`) → assert `"conclusion"`.
- **D-06:** `test_graph_wiring.py` also tests `build_graph()` compiles without raising (smoke test) and that `build_graph(checkpointer=MemorySaver())` returns a compiled graph.

### Output Surface Tests (Phase 7 coverage)
- **D-07:** `test_output_surface.py` tests `cli._print_human()` using pytest's `capsys` fixture. Two cases: (a) verdict dict with `verdict_label_vi` present → assert `"Thật"` or `"Giả"` in captured stdout; (b) verdict dict without `verdict_label_vi` → assert old-format label appears (fallback, Phase 7 D-02).
- **D-08:** `test_output_surface.py` tests `run_fact_check()` top-level field promotion (Phase 7 D-04): mock `build_graph().invoke()` to return a verdict with `verdict_binary`/`verdict_label_vi`, call `run_fact_check("statement")`, assert both fields are present at the top level of the returned dict.
- **D-09:** `test_output_surface.py` tests `mcp_server.fact_check()` response dict: mock `build_graph().invoke()`, call the tool function, assert `"verdict_binary"` and `"verdict_label_vi"` keys exist in the returned dict (Phase 7 D-05).

### Claude's Discretion
- Mock return values for agent patches in TEST-06 — planner/executor chooses minimal state dicts (e.g., search returns `{"evidence": [], "evidence_graph": None}`, conclusion returns `{"verdict": {"label": "UNVERIFIED", "verdict_binary": "FAKE", "verdict_label_vi": "Giả"}}`).
- Whether `test_graph_wiring.py` and `test_output_surface.py` are separate files or merged — one-file-per-concern preferred for readability.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §Tests (TEST-01..TEST-06) — exact requirement definitions for each test category
- `.planning/ROADMAP.md` §Phase 8 — success criteria and plan structure (08-01, 08-02)

### Prior Phase Context (what's being tested)
- `.planning/phases/06-langgraph-wiring/06-CONTEXT.md` — D-03 (`build_graph(checkpointer=None)` signature), D-07 (`route_after_verify` reading `reliability_signal`); these are the behaviors graph wiring tests must verify
- `.planning/phases/07-output-surface/07-CONTEXT.md` — D-01 (`_print_human` new format), D-02 (fallback format), D-04 (`run_fact_check()` top-level promotion), D-05 (MCP response top-level fields); these are the exact behaviors output surface tests must verify

### Existing Test Files (already satisfy TEST-01..05 — do NOT rewrite these)
- `tests/factcheck_agents/test_source_tier.py` — TEST-01: source-tier classification
- `tests/factcheck_agents/test_evidence_graph.py` — TEST-02: evidence graph construction
- `tests/factcheck_agents/test_verify_agent.py` — TEST-03: reliability_signal computation + verify_agent integration
- `tests/factcheck_agents/test_conclusion_agent.py` — TEST-04 + TEST-05: binary verdict mapping, verdict_label_vi, fallback verdict, LLM verdict, conflict detection
- `tests/factcheck_agents/test_search_agent.py` — Phase 2 search agent coverage
- `tests/factcheck_agents/test_social_search_agent.py` — Phase 4 social search coverage

### New Files to Create
- `tests/factcheck_agents/test_graceful_degrade.py` — TEST-06 (full pipeline integration)
- `tests/factcheck_agents/test_graph_wiring.py` — Phase 6 graph routing unit tests
- `tests/factcheck_agents/test_output_surface.py` — Phase 7 output surface unit tests

### Existing Code to Read Before Implementing
- `factcheck_agents/graph.py` — `build_graph()`, `route_after_verify` — must be importable for unit testing
- `factcheck_agents/cli.py` — `_print_human()` — needs to be importable directly for capsys testing
- `factcheck_agents/__init__.py` — `run_fact_check()` — top-level promotion logic (D-04)
- `factcheck_agents/mcp_server.py` — `fact_check()` tool function (D-05)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/factcheck_agents/test_verify_agent.py` `_make_state()` helper — minimal state dict builder; use the same pattern in `test_graceful_degrade.py`
- `tests/factcheck_agents/test_conclusion_agent.py` `_make_state()` and `_build_graph()` — reusable patterns for constructing EvidenceGraph and state
- `@patch('factcheck_agents.agents.verify_agent._run_coolant')` pattern from `test_verify_agent.py` — established mock path format for this package

### Established Patterns
- `unittest.mock.patch` as decorator or context manager — used throughout all existing factcheck_agents tests; follow the same style
- `pytest.mark.parametrize` — used in `test_source_tier.py`; use for `route_after_verify` True/False/None cases
- `from factcheck_agents.agents.X import X` import style — one import per function under test
- `MemorySaver` from `langgraph.checkpoint.memory` — use for `build_graph(checkpointer=MemorySaver())` in all graph-level tests to avoid sqlite extra requirement

### Integration Points
- `build_graph()` in `factcheck_agents/graph.py` — central integration point; passing `MemorySaver()` explicitly avoids env var / sqlite interaction in tests
- `factcheck_agents/agents/__init__.py` exports `search_agent`, `verify_agent`, `social_search_agent`, `conclusion_agent` — patch these at their canonical path when mocking for TEST-06
- `pytest.ini` `addopts = -ra -v --disable-warnings` and `testpaths = tests` — new files in `tests/factcheck_agents/` are auto-discovered; no pytest.ini changes needed

</code_context>

<specifics>
## Specific Ideas

- TEST-06 mock setup: the conclusion_agent mock must return `{"verdict": {"label": "UNVERIFIED", "confidence": 0.0, "verdict_binary": "FAKE", "verdict_label_vi": "Giả", "rationale": "", "citations": [], "recommendation": ""}}` — this satisfies the `no exception + verdict is dict` assertion.
- `route_after_verify` parametrize: `@pytest.mark.parametrize("signal,expected", [(True, "social_search"), (False, "conclusion"), (None, "conclusion")])`.
- `_print_human` capsys test: call `_print_human({"label": "TRUE", "confidence": 0.9, "verdict_label_vi": "Thật", ...})` then `captured = capsys.readouterr(); assert "Thật" in captured.out`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 8-Tests*
*Context gathered: 2026-07-27*
