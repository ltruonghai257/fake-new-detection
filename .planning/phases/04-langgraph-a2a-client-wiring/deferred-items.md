# Deferred Items — Phase 04

Out-of-scope discoveries logged per executor scope boundary (not fixed in this phase).

## Pre-existing test failures (confirmed at baseline HEAD~5, before Phase 04 changes)

| Test | Failure | Status |
| ---- | ------- | ------ |
| `tests/factcheck_agents/test_agreement_gate.py::test_agreement_unavailable_model_treated_as_zero` | `assert result["agreement_score"] > 0` → 0.0 | open |
| `tests/factcheck_agents/test_conclusion_agent.py::TestBinaryHelpers::test_map_to_binary_unverified_maps_to_fake` | `('NEI', 'Chưa xác thực') != ('FAKE', 'Giả')` — UNVERIFIED maps to NEI, not FAKE | open |
| `tests/factcheck_agents/test_conclusion_agent.py::TestLlmVerdict::test_llm_unverified_maps_to_fake` | `'NEI' != 'FAKE'` | open |
| `tests/factcheck_agents/test_debate_pipeline_integration.py::test_worldcup_claim` | verdict `'NEI' not in {'FAKE', 'REAL'}` | open |
| `tests/factcheck_agents/test_debate_pipeline_integration.py::test_nei_short_circuit` | `'UNVERIFIED' != 'NEI'` | open |

These look like label-mapping expectation drift (NEI/UNVERIFIED vocabulary vs test
expectations), pre-dating Phase 04. Plan 04-02 acceptance criteria assume "83 tests
pass" — the suite now collects 104 tests with these 5 pre-existing failures, so the
04-02 criteria are interpreted as "no NEW failures introduced by Phase 04".
