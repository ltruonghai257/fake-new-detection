# Phase 1 Summary: Debate Pipeline

**Status**: Complete
**Execution Date**: 2026-08-03
**Waves**: 5/5 executed
**Plans**: 5/5 completed

## Overview

Phase 1 successfully implemented the complete debate-based verification pipeline for the factcheck_agents v3.0 milestone. All 18 requirements (EVRET-01..04, RERANK-01..02, SOCLOOP-01..03, AGREE-01..03, DEBATE-01..03, JUDGE-01..03) were satisfied across 5 waves of execution.

## Waves Executed

### Wave 1 (01-01): State Extensions, Config Settings, and Dependency Setup
**Commit**: ca1c2f1
- Extended `FactCheckState` with 10 M2 fields (evidence_real, evidence_fake, evidence_social, social_loop_fired, request_id, consistency_score, agreement_score, debate_turns, debate_exit_reason, weight_breakdown)
- Added 5 M2 settings to config.py (agreement_threshold, max_debate_rounds, google_factcheck_api_key, social_loop_min_count, social_loop_min_credibility)
- Added rank_bm25>=0.2.2 to pyproject.toml
- Updated initial_state() with M2 defaults (UUID generation, empty lists, flags)
- **Verification**: 113 tests passed, all acceptance criteria met

### Wave 2 (01-02): Evidence Retrieval Agents and BM25+Embedding Reranker
**Commit**: 25a209a
- Created `factcheck_agents/reranker.py` with BM25+PhoBERT CLS embedding reranker, greedy 256-token fill, consistency_score computation, graceful degradation
- Created `factcheck_agents/agents/real_source_agent.py` for trusted domain search (vnexpress.net, tuoitre.vn, thanhnien.vn, ttxvn.gov.vn, vtv.vn, dantri.com.vn)
- Created `factcheck_agents/agents/fake_source_agent.py` for tingia.gov.vn + Google Fact Check API integration
- Exported new agents from agents/__init__.py
- Created `tests/factcheck_agents/test_reranker.py` with 5 unit tests (recall@k, consistency_score floor, empty pool, single-item, field name verification)
- **Verification**: 118 tests passed, all acceptance criteria met

### Wave 3 (01-03): Social Loop Agent, Agreement Gate, and Graph Routing Helpers
**Commit**: a5d620b
- Created `factcheck_agents/agents/social_loop_agent.py` targeting tiktok.com + flagged_domains, sets social_loop_fired=True
- Created `factcheck_agents/agents/agreement_gate.py` with 0.30/0.30/0.40 weighted formula and routing function
- Added routing helpers to graph.py (route_nei_check, route_social_loop) without modifying build_graph()
- Created `tests/factcheck_agents/test_social_loop_agent.py` with 5 tests (fire-once guard, domain targeting, exception handling, EVRET-03 separation)
- Created `tests/factcheck_agents/test_agreement_gate.py` with 6 tests (weighted formula, NEI forcing zero, credibility floor, unavailable models, routing)
- **Verification**: 129 tests passed, all acceptance criteria met

### Wave 4 (01-04): Debate Node, Judge Agent, and M2 Graph Topology
**Commit**: 5c7b586
- Created `factcheck_agents/agents/debate_node.py` with bounded for-loop debate, real/fake advocates, JSONL logging to logs/debates/<request_id>.jsonl, no-LLM guard
- Created `factcheck_agents/agents/judge_agent.py` with 1-5 dimension scoring, NEI short-circuit, weighted verdict (30/30/40), JSON logging to logs/verdicts/<request_id>.json
- Added `build_debate_graph()` to graph.py with full M2 topology (static fan-out, NEI gate, reranker, social_loop, verify, agreement_gate, debate, judge)
- Exported new agents from agents/__init__.py (social_loop_agent, debate_node, judge_agent)
- **Verification**: 129 tests passed, both build_graph() and build_debate_graph() compile successfully

### Wave 5 (01-05): Integration Tests — Full M2 Pipeline
**Commit**: f39c303
- Created `tests/factcheck_agents/test_debate_pipeline_integration.py` with 5 integration tests:
  - test_worldcup_claim: Full pipeline with Vietnamese claim
  - test_vaccine_claim: Second Vietnamese claim test
  - test_nei_short_circuit: EVRET-04 NEI path verification
  - test_logs_dirs_exist: Directory creation verification
  - test_no_state_collision_between_runs: MemorySaver isolation test
- Mock setup: 5 agents mocked at factcheck_agents.graph namespace, agreement_gate and debate_node run with real logic
- **Verification**: 134 tests passed (5 new + 129 existing), all acceptance criteria met

## Requirements Satisfied

All 18 Phase 1 requirements were successfully implemented:

- **EVRET-01..04**: Dual-source evidence retrieval (real/fake agents), NEI gate, separation of evidence fields
- **RERANK-01..02**: BM25+embedding reranking with 256-token budget, recall@k verification
- **SOCLOOP-01..03**: One-shot social search, fire-once guard, evidence field separation
- **AGREE-01..03**: Weighted agreement formula (0.30/0.30/0.40), high-agreement skip, logging
- **DEBATE-01..03**: Bounded advocate debate, JSONL turn logging, no-LLM guard
- **JUDGE-01..03**: 1-5 dimension scoring, weighted verdict, structured JSON logging

## Test Coverage

- **Total test count**: 134 tests (5 new integration tests + 129 existing)
- **New test files**: 3 (test_reranker.py, test_social_loop_agent.py, test_agreement_gate.py, test_debate_pipeline_integration.py)
- **Regression verification**: All existing tests continue to pass
- **Integration coverage**: End-to-end pipeline verified with 2 Vietnamese claims + NEI path

## Key Architectural Decisions

1. **Static fan-out**: Used native LangGraph fan-out (not Send API) for real_source + fake_source parallelism
2. **BM25+PhoBERT reranking**: Linear combination (α=0.5) with greedy token budgeting, BM25-only fallback when PhoBERT unavailable
3. **Evidence field separation**: evidence_real, evidence_fake, evidence_social remain separate until reranker merges to unified evidence field
4. **Agreement gate formula**: 0.30 × phobert_confidence + 0.30 × coolant_confidence + 0.40 × evidence_credibility
5. **Graph topology**: NEI gate → reranker → social_loop (conditional) → verify → agreement_gate → debate (conditional) → judge
6. **Backward compatibility**: build_graph() unchanged for M1, build_debate_graph() added for M2
7. **Graceful degradation**: All nodes have no-LLM guards and exception handling to prevent pipeline crashes

## Artifacts Created

**New Source Files**:
- factcheck_agents/reranker.py (BM25+embedding reranker)
- factcheck_agents/agents/real_source_agent.py (trusted domain search)
- factcheck_agents/agents/fake_source_agent.py (fact-checking sources)
- factcheck_agents/agents/social_loop_agent.py (weak-evidence social search)
- factcheck_agents/agents/agreement_gate.py (agreement computation + routing)
- factcheck_agents/agents/debate_node.py (bounded advocate debate)
- factcheck_agents/agents/judge_agent.py (weighted judge with dimension scoring)

**Modified Source Files**:
- factcheck_agents/state.py (10 new M2 fields)
- factcheck_agents/config.py (5 new M2 settings)
- factcheck_agents/graph.py (routing helpers + build_debate_graph)
- factcheck_agents/agents/__init__.py (exported 7 new agents/functions)
- pyproject.toml (added rank_bm25>=0.2.2)

**New Test Files**:
- tests/factcheck_agents/test_reranker.py (5 tests)
- tests/factcheck_agents/test_social_loop_agent.py (5 tests)
- tests/factcheck_agents/test_agreement_gate.py (6 tests)
- tests/factcheck_agents/test_debate_pipeline_integration.py (5 tests)

**Log Directories** (created at runtime):
- logs/debates/ (JSONL turn-by-turn debate logs)
- logs/verdicts/ (JSON verdict logs with weight breakdown)

## Commits

1. ca1c2f1 - gsd: phase 01 wave 1 complete — state, config, deps
2. 25a209a - gsd: phase 01 wave 2 complete — evidence agents + reranker
3. a5d620b - gsd: phase 01 wave 3 complete — social loop + agreement gate
4. 5c7b586 - gsd: phase 01 wave 4 complete — debate node + judge + M2 graph
5. f39c303 - gsd: phase 01 wave 5 complete — integration tests

## Next Steps

Phase 1 is complete. The project is now ready for Phase 2 (Demo App) which will:
- Build a FastAPI SSE backend for live debate streaming
- Create a React/Vite/TypeScript frontend with Vietnamese UI
- Implement turn-by-turn debate visualization with alternating chat bubbles
- Add verdict card with 30/30/40 weight breakdown
- Support log file downloads (JSONL + JSON)

**Dependencies**: Phase 2 depends on the stable `build_debate_graph()` function now available in graph.py.