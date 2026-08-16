# Deferred Items — Phase 04

Out-of-scope discoveries logged per executor scope boundary (not fixed in this phase).

## Pre-existing test failures (confirmed at baseline HEAD~5, before Phase 04 changes)

| Test                                                                                                             | Failure                                                                          | Status |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------ |
| `tests/factcheck_agents/test_agreement_gate.py::test_agreement_unavailable_model_treated_as_zero`                | `assert result["agreement_score"] > 0` → 0.0                                     | open   |
| `tests/factcheck_agents/test_conclusion_agent.py::TestBinaryHelpers::test_map_to_binary_unverified_maps_to_fake` | `('NEI', 'Chưa xác thực') != ('FAKE', 'Giả')` — UNVERIFIED maps to NEI, not FAKE | open   |
| `tests/factcheck_agents/test_conclusion_agent.py::TestLlmVerdict::test_llm_unverified_maps_to_fake`              | `'NEI' != 'FAKE'`                                                                | open   |
| `tests/factcheck_agents/test_debate_pipeline_integration.py::test_worldcup_claim`                                | verdict `'NEI' not in {'FAKE', 'REAL'}`                                          | open   |
| `tests/factcheck_agents/test_debate_pipeline_integration.py::test_nei_short_circuit`                             | `'UNVERIFIED' != 'NEI'`                                                          | open   |

These look like label-mapping expectation drift (NEI/UNVERIFIED vocabulary vs test
expectations), pre-dating Phase 04. Plan 04-02 acceptance criteria assume "83 tests
pass" — the suite now collects 104 tests with these 5 pre-existing failures, so the
04-02 criteria are interpreted as "no NEW failures introduced by Phase 04".

## Environment drift: langgraph checkpointer (fixed in scope, impacts remain)

The langgraph upgrade in this environment added two production-breaking behaviors
(confirmed at baseline HEAD~8, pre-dating v3.1):

-   `graph.invoke()` without `thread_id` config raises ValueError. Fixed in `cli.py`
    (Phase 04 deviation). `run_fact_check()` (`factcheck_agents/__init__.py:36`) and
    `mcp_server.py:44` still invoke without `thread_id` — both crash on any run.
    **open** — Phase 5 or maintenance.
-   Checkpointer msgpack serialization rejects `EvidenceGraph` instances. Fixed in
    `graph_utils.py` (`_asdict()`/`graph_data` round-trip, Phase 04 deviation). Note:
    langgraph warns this unregistered type will be blocked in a future version
    (`LANGGRAPH_STRICT_MSGPACK=true` blocks now).

## Plan defects found during execution (deviations, see SUMMARYs)

-   RESEARCH Pattern 5 (04-RESEARCH.md) assumed patching `factcheck_agents.a2a_client.*`
    intercepts graph node calls. With `from .a2a_client import X` bindings it does NOT;
    `factcheck_agents.graph.*` remains the effective patch target. Tests were kept
    (or reverted to) graph-namespace patching.
-   Plan's Task 04-01-05 verification commands used invalid one-liner Python
    (`@decorator; def` / `with` in `;` chains) — re-run multi-line, assertions identical.
