# fake-new-detection / factcheck_agents

## What This Is

A multi-agent LangGraph module (`factcheck_agents/`) that fact-checks Vietnamese news statements
using web evidence and two trained models (PhoBERT ViFactCheck + COOLANT). Agents share a single
typed state object and run in sequence — inspired by TradingAgents' shared-state design.
The module is fully decoupled from the training pipeline; model checkpoints are lazy-loaded and
every failure degrades gracefully.

## Core Value

A user submits a Vietnamese claim and gets back a binary verdict — **Thật** or **Giả** — with a
Vietnamese-language rationale and citations they can verify, even when model checkpoints are missing.

## Previous Milestone: v3.0 Debate-Based Verification Pipeline and Demo App ✓

**Goal:** Replace the evidence-overrides-models correctness bug with a debate-based multi-agent
architecture and ship a local thesis defense demo web app with live streaming.

**Shipped:** Dual-source evidence retrieval, BM25+embedding reranking, social loop, agreement gate,
bounded advocate debate with JSONL logging, weighted judge (30/30/40), FastAPI + React/Vite/TypeScript
demo app with live SSE debate streaming and Vietnamese verdict card.

---

## Current Milestone: v3.1 A2A Protocol Integration

**Goal:** Refactor all factcheck agents to conform to the Google Agent2Agent (A2A) protocol
(`a2a-sdk`), running as independent HTTP services, while keeping LangGraph as the routing
orchestrator and preserving the demo app's SSE streaming.

**Target features:**

**Phase 3 complete (2026-08-15):** all 10 agents wrapped as A2A services
(`a2a_sdk`-based `AgentExecutor` handlers on ports 9001–9010, Agent Cards at
`/.well-known/agent.json`, start/stop/smoke scripts shipped). `debate_node`
split into single-turn `real_advocate`/`fake_advocate` services.

**Phase 4 complete (2026-08-17):** LangGraph nodes now invoke every agent via
`a2a_client.py` over A2A HTTP (sync `httpx.Client` bridge, `AgentUnavailableError`
→ per-agent degrade diffs, partial-debate semantics in `debate_node`). Verified
end-to-end: CLI smoke test produced a live verdict in 58s with all servers up,
and a graceful UNVERIFIED degrade with all servers down. Phase 5 (Demo App +
Tests) is next.

-   All 10 agents wrapped as A2A `TaskHandler`s (`a2a-sdk[http-server,fastapi]`): `search_agent`, `evaluate_agent`, `real_source_agent`, `fake_source_agent`, `social_loop_agent`, `agreement_gate`, `real_advocate`, `fake_advocate`, `judge_agent`, `conclusion_agent`
-   Each agent served by its own uvicorn HTTP server on a dedicated port (9001–9010) in local dev; `scripts/start_agents.sh` starts all; `scripts/stop_agents.sh` stops all
-   Agent Card (`/.well-known/agent.json`) per agent for standard A2A service discovery
-   LangGraph edges refactored: graph nodes call `A2AClient` over HTTP instead of invoking agent functions directly; LangGraph retained for routing/conditional edges only
-   Demo app FastAPI SSE bridge updated to call A2A agent HTTP endpoints; streaming still delivered turn-by-turn to the React frontend unchanged
-   No breaking changes to existing CLI, Python API (`run_fact_check()`), or MCP server interfaces

## Requirements

### Validated

<!-- Shipped v1.0 baseline — confirmed working -->

-   ✓ **CLI-01**: User can fact-check a statement via `python -m factcheck_agents.cli` — v1.0
-   ✓ **CLI-02**: User can supply an optional image path to enable COOLANT — v1.0
-   ✓ **API-01**: `run_fact_check(statement)` Python API returns verdict + rationale + citations — v1.0
-   ✓ **MCP-01**: MCP server exposes `fact_check`, `search_evidence`, `evaluate_statement` tools — v1.0
-   ✓ **SEARCH-01**: Search agent drafts queries (LLM or heuristic fallback) and retrieves web evidence via Tavily/Google CSE — v1.0
-   ✓ **EVAL-01**: Evaluate agent runs PhoBERT ViFactCheck (statement + evidence → SUPPORTED/REFUTED/NEI) — v1.0
-   ✓ **EVAL-02**: Evaluate agent runs COOLANT (image → REAL/FAKE) only when image supplied — v1.0
-   ✓ **EVAL-03**: Missing model checkpoints produce `unavailable` result; pipeline never crashes — v1.0
-   ✓ **CONCL-01**: Conclusion agent fuses model verdicts + evidence into 4-class verdict (TRUE/FALSE/MISLEADING/UNVERIFIED) — v1.0
-   ✓ **CONCL-02**: Rule-based fallback verdict when LLM is unavailable — v1.0
-   ✓ **CFG-01**: All settings come from env vars (no hardcoded secrets) — v1.0
-   ✓ **A2A-01**: `a2a-sdk[http-server,fastapi]` added to `factcheck_agents/pyproject.toml` (or requirements); all 10 agent modules implement `TaskHandler` protocol — v3.1 Phase 3, validated 2026-08-15
-   ✓ **A2A-02**: Each agent exposes `GET /.well-known/agent.json` (A2A Agent Card) with name, description, skills, and port — v3.1 Phase 3, validated 2026-08-15
-   ✓ **A2A-03**: `scripts/start_agents.sh` starts all 10 uvicorn servers (ports 9001–9010) and writes a PID file; `scripts/stop_agents.sh` stops them cleanly — v3.1 Phase 3, validated 2026-08-15
-   ✓ **A2A-04**: `factcheck_agents/a2a_client.py` wraps `A2AClient` calls; LangGraph nodes import this module instead of agent functions; state passing uses A2A `Task` messages — v3.1 Phase 4, validated 2026-08-17
-   ✓ **A2A-05**: LangGraph `graph.py` updated — `build_debate_graph()` and `build_graph()` call A2A clients; conditional routing edges unchanged — v3.1 Phase 4, validated 2026-08-17
-   ✓ **A2A-05b**: `debate_node` handles per-advocate `AgentUnavailableError` with partial-debate semantics (available advocate continues; both down → `agent_unavailable`) — v3.1 Phase 4, validated 2026-08-17

### Active

<!-- v3.1 scope — A2A Protocol Integration -->

-   [ ] **A2A-06**: `demo_app/backend/streaming.py` updated to call A2A agent HTTP endpoints; SSE `turn_start`/`chunk`/`turn_end` events unchanged for the React frontend
-   [ ] **A2A-07**: Unit tests updated: each agent tested via its A2A HTTP interface (spin up in-process uvicorn); existing graph integration tests adapted for A2A client calls
-   [ ] **A2A-08**: CLI (`cli.py`), Python API (`run_fact_check()`), and MCP server (`mcp_server.py`) remain externally unchanged; they call `build_debate_graph()` as before

### Out of Scope

-   Modifying PhoBERT/COOLANT model weights, architecture, or training code
-   Removing the existing source-tier system (extend it for evidence-credibility in JUDGE-02)
-   Public deployment of the demo app
-   Paid X/Twitter API or Meta Graph API
-   Modifying `training/`, model checkpoints, or notebooks
-   gRPC or cloud A2A deployment (local HTTP only for v3.1)
-   Changing the React frontend — only the backend SSE bridge is updated

## Context

-   **Stack**: Python, LangGraph, PhoBERT (vinai/phobert-base-v2 via HuggingFace), COOLANT (custom checkpoint), Tavily/Google CSE, FastMCP, FastAPI, React/Vite/TypeScript (demo_app/), `a2a-sdk` (Google Agent2Agent Protocol)
-   **Architecture reference**: TradingAgents (shared state, Bull/Bear debate → judge), ViFactCheck AAAI 2025, Debate-to-Detect / TruEDebate / DebateCV (multi-dimensional argument scoring)
-   **Evidence graph pattern**: plain Python networkx structure, built once, queried downstream
-   **Binary mapping default**: MISLEADING → FAKE, UNVERIFIED → FAKE ("not verifiably real" cannot be reported as Thật)
-   **Debate weights**: PhoBERT 30% + COOLANT 30% + evidence-credibility 40% — flagged as tunable defaults
-   **Graceful degrade**: a missing checkpoint marks that sub-verdict `unavailable`; always preserved

## Constraints

-   **Scope**: Work only inside `factcheck_agents/` and `tests/` — never touch `training/` or notebooks
-   **Dependencies**: No new paid APIs without explicit confirmation; prefer packages already in `requirements.txt`
-   **Compatibility**: New binary verdict fields must be additive (no breaking change to existing callers)
-   **Language**: All user-facing output in Vietnamese; internal field/enum names stay English for API stability
-   **Evidence graph lib**: `networkx` is acceptable; also a plain `dict`-of-dicts is sufficient — keep it minimal

## Key Decisions

| Decision                                                 | Rationale                                                                                     | Outcome        |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------- |
| LangGraph for orchestration                              | Shared-state, conditional edges, composable nodes                                             | ✓ Good         |
| Lazy model loading with `lru_cache`                      | Avoids import-time torch overhead; degrades when checkpoint missing                           | ✓ Good         |
| `unavailable` result (not raise) for missing checkpoints | Pipeline must always complete; callers check `available` field                                | ✓ Good         |
| Tavily primary / Google CSE fallback                     | Tavily returns cleaner LLM-ready snippets; CSE as resilience fallback                         | ✓ Good         |
| 4-class verdict → 2-class binary (v2.0)                  | MISLEADING/UNVERIFIED cannot be reported as Thật; binary is user-facing                       | ✓ Shipped v2.0 |
| Evidence graph as plain Python structure (v2.0)          | Avoid new heavy deps; "build once" pattern without npm                                        | ✓ Shipped v2.0 |
| Debate architecture over single-pass verdict (v3.0)      | Single-pass evidence silently overrides model predictions; debate forces explicit weighting   | ✓ Shipped v3.0 |
| 30/30/40 weight split (v3.0)                             | Equal model weight, evidence-credibility plurality; tunable default in STATE.md               | ✓ Shipped v3.0 |
| Google Fact Check API stubbed until key provided (v3.0)  | Avoid silent failures; user must opt in explicitly                                            | ✓ Shipped v3.0 |
| A2A protocol for agent communication (v3.1)              | Standardized HTTP-based agent interface; each agent becomes independently testable/deployable | ✓ Shipped v3.1 |
| LangGraph retained as routing-only orchestrator (v3.1)   | A2A handles message passing; LangGraph handles conditional graph edges — minimal coupling     | ✓ Shipped v3.1 |
| Sync httpx.Client bridge for graph → agent calls (v3.1)  | Graph nodes are sync; per-call client avoids event-loop/async churn — zero caller changes     | ✓ Shipped v3.1 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):

1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):

1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---

_Last updated: 2026-08-17 — v3.1 Phase 4 (LangGraph → A2A Client Wiring) complete_
