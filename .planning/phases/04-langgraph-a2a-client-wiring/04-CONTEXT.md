# Phase 4: LangGraph → A2A Client Wiring - Context

**Gathered:** 2026-08-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `factcheck_agents/a2a_client.py` with typed wrapper functions (one per A2A agent), refactor `graph.py` node functions to call `a2a_client.*` instead of importing agent functions directly, and handle `AgentUnavailableError` gracefully in every node. The 4 non-wrapped nodes (`verify_agent`, `reranker`, `expert_agent`, `social_search_agent`) stay as local imports — only the 10 Phase 3 A2A services (ports 9001–9010) get wired through HTTP. `build_graph()` and `build_debate_graph()` signatures and return types unchanged.

**Not in scope:** Demo app SSE bridge update (Phase 5), agent HTTP-level tests (Phase 5), session-scoped agent-server test fixtures (Phase 5).
</domain>

<decisions>
## Implementation Decisions

### Sync Bridge — Async Client, Sync Graph

- **D-01:** `a2a_client` exposes sync wrapper functions per agent (using `httpx.Client` or `asyncio.run` shim internally). Graph nodes and all entry points (`run_fact_check`, `cli.py`, `mcp_server.py`, `demo_app/backend/streaming.py`) remain sync `.invoke()` — zero caller changes. — **Reversibility:** costly — switching to `async def` nodes + `ainvoke` would touch all 9 graph nodes, 4 entry points, and the demo streaming bridge.

- **D-02:** Each wrapper takes `FactCheckState` → returns a state diff `dict`, mirroring the existing agent function signature. "Typed" means the wrapper validates/deserializes the `TaskResult.output` back to Python types matching the LangGraph state fields (`Evidence`, `DebateTurn`, etc.).

### Non-JSON-Safe State Objects — evidence_graph

- **D-03:** The `EvidenceGraph` object in `search_agent`'s diff is NOT transmitted over the wire (the `a2a_server.serialize_state` repr-fallback turns it into a string that would crash `conclusion_agent`). The search client wrapper reconstructs `EvidenceGraph.build_from_evidence(evidence)` locally from the returned evidence list after deserializing the diff. No handler-side or serialization changes needed. — **Reversibility:** reversible — local change in the client wrapper only.

### Graceful Degrade — Shared Decorator

- **D-04:** `@degrade_on_unavailable` decorator on each `a2a_client` wrapper function. On `AgentUnavailableError`, the decorator returns a degrade diff for that agent (empty/None values for the agent's output fields + a `messages` note like `"[Search] agent unavailable — degraded"`). The per-agent degrade diffs are defined in a shared helper in `a2a_client.py`. Each graph node catches nothing — the decorator handles it. — **Reversibility:** costly — touching all 9 wrappers to change the degrade pattern.

- **D-05:** Debate advocates: **partial debate**. If only one advocate is reachable, the debate loop runs one-sided turns (the available advocate argues; the unavailable advocate's turns are skipped). `debate_exit_reason` marks which side was unavailable. Judge receives partial argument scores. If both advocates are down, `debate_exit_reason='agent_unavailable'` and judge runs without argument quality signal (confidence capped at 0.7 per JUDGE-02). — **Reversibility:** reversible — inside `debate_node` only.

### Timeout & Retry Policy

- **D-06:** Per-agent httpx timeouts based on agent type:
  - LLM agents (search, real_source, fake_source, real_advocate, fake_advocate, judge, conclusion) = **120 s**
  - Calculation-only (agreement_gate) = **30 s**
  - Web-search (social_loop) = **30 s**
  - Model-inference (evaluate_agent) = **60 s**
  - Global override via `A2A_CLIENT_TIMEOUT` env var; per-agent override via `A2A_CLIENT_TIMEOUT_<NAME>` env vars.

- **D-07:** **No automatic retry.** Connection errors → `AgentUnavailableError` immediately (fail-fast). The debate loop handles individual advocate failures via D-05 partial-debate semantics.

### Graph-Level Test Adaptation

- **D-08:** `tests/factcheck_agents/test_graceful_degrade.py` and `tests/factcheck_agents/test_debate_pipeline_integration.py` patch targets updated from `factcheck_agents.graph.<agent>` to `factcheck_agents.a2a_client.<agent>`. Minimal edits (~10 lines each). Other 81+ agent-level unit tests unchanged.

### Claude's Discretion

- evidence_graph rebuild: in the client wrapper (not handler-side or serialization change) — local, no contract changes
- Per-agent timeout values based on agent type (LLM 120s vs calculation 30s vs model-inference 60s)
- `messages` tuple→list normalization: handled in the client wrapper on return (lists unpack identically to tuples in `for role, content in messages`; no downstream consumers check `isinstance(msg, tuple)`)
- `evaluate_agent` wrapper exists for API completeness (port 9002) even though no graph node calls it — MCP uses `evaluate_agent` in-process
- Partial debate: the loop retries the available advocate; unavailable side turns are marked with `{"agent": "<side>", "error": "agent_unavailable"}` in `debate_turns`
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope & Requirements
- `.planning/ROADMAP.md` §"Phase 4: LangGraph → A2A Client Wiring" — success criteria (5 bullets), plan breakdown (04-01, 04-02), dependency on Phase 3
- `.planning/REQUIREMENTS.md` — A2A-04 (a2a_client module), A2A-05 (graph refactor), A2A-05b (AgentUnavailableError degrade)
- `.planning/STATE.md` — port assignments (9001–9010), tunable defaults (agreement threshold, weights, max_rounds)
- `.planning/PROJECT.md` — v3.1 milestone definition, target features, out-of-scope list

### Phase 3 Decisions (locked — do not re-litigate)
- `.planning/phases/03-a2a-agent-wrappers-launch-scripts/03-CONTEXT.md` — D-01..D-15 (Task input/output contract, debate_role encoding, serialization helpers, BaseTaskHandler flow, AgentCardConfig, launch script behavior)

### Existing Code (MUST read before writing a2a_client.py or refactoring graph.py)
- `factcheck_agents/a2a_server.py` — `serialize_state`, `deserialize_state`, `BaseTaskHandler.execute()` flow, `AgentCardConfig`, `create_app`, `run_server`; the Task lifecycle (D-01: full state → D-02: diff → D-04: failed on exception)
- `factcheck_agents/config.py` — `Settings` dataclass, `a2a_ports()` mapping (agent name → port), env-var patterns (add A2A_CLIENT_TIMEOUT* here)
- `factcheck_agents/graph.py` — `build_graph()` (M1: search→verify→social_search→conclusion), `build_debate_graph()` (M2: fan-out→nei_gate→reranker→social_loop→verify→agreement_gate→debate→judge→expert), `debate_node()` orchestration loop, routing functions
- `factcheck_agents/state.py` — `FactCheckState` TypedDict (all fields; `evidence_graph` is an in-memory object)
- `factcheck_agents/agents/search_agent.py` — `search_agent(state)` → returns `evidence`, `evidence_graph`, `search_queries`, `claim_variants`, `messages`
- `factcheck_agents/agents/conclusion_agent.py` — `conclusion_agent(state)` → reads `evidence_graph` (unguarded `_has_cross_source_conflict` call at line 179)
- `factcheck_agents/agents/real_source_agent.py` — function signature and return type
- `factcheck_agents/agents/fake_source_agent.py` — function signature and return type
- `factcheck_agents/agents/social_loop_agent.py` — function signature and return type
- `factcheck_agents/agents/agreement_gate.py` — function signature + `route_after_agreement`
- `factcheck_agents/agents/real_advocate.py` — `real_advocate(state)` → `{"debate_turn": ..., "messages": [...]}`; single-turn; reads `debate_role`
- `factcheck_agents/agents/fake_advocate.py` — same shape as real_advocate
- `factcheck_agents/agents/judge_agent.py` — function signature; produces scores only (expert_agent produces final verdict)
- `factcheck_agents/agents/evaluate_agent.py` — function signature; wrapped for completeness (MCP uses in-process)

### Test Files to Update
- `tests/factcheck_agents/test_graceful_degrade.py` — patches `factcheck_agents.graph.<agent>` (update to `a2a_client.<agent>`)
- `tests/factcheck_agents/test_debate_pipeline_integration.py` — same patch-target update
</canonical_refs>

<code_context>
## Existing Code Insights

### What Stays Local (no A2A — direct imports unchanged)
- `verify_agent` (PhoBERT+COOLANT model inference) — no wrapper exists, stays as direct import in graph.py
- `reranker` (BM25+PhoBERT embedding rerank) — local, no wrapper
- `expert_agent` (final verdict fusion after judge) — local, no wrapper
- `social_search_agent` (M1 twitter/facebook search) — local, M1-only

### What Becomes A2A (HTTP calls)
| Graph | Node | A2A Agent | Port |
|-------|------|-----------|------|
| M1 `build_graph()` | search | search_agent | 9001 |
| M1 `build_graph()` | conclusion | conclusion_agent | 9010 |
| M2 `build_debate_graph()` | real_source | real_source_agent | 9003 |
| M2 `build_debate_graph()` | fake_source | fake_source_agent | 9004 |
| M2 `build_debate_graph()` | social_loop | social_loop_agent | 9005 |
| M2 `build_debate_graph()` | agreement_gate | agreement_gate | 9006 |
| M2 `build_debate_graph()` | debate (orchestrator) | real_advocate + fake_advocate | 9007 + 9008 |
| M2 `build_debate_graph()` | judge | judge_agent | 9009 |
| — (unused by graph) | — | evaluate_agent | 9002 |

### Reusable Assets
- `a2a_ports()` in `config.py` — maps agent name → port; use directly for client base URL construction
- `serialize_state` / `deserialize_state` in `a2a_server.py` — reuse for constructing `Task` payloads in the client
- `EvidenceGraph.build_from_evidence()` — call in the search client wrapper to rebuild the in-memory graph locally (D-03)
- `a2a.types.Task`, `a2a.types.TaskState` — A2A SDK types for constructing Task messages in the client
- `a2a.client.A2AClient` — SDK-provided HTTP client (or use raw `httpx` for simpler sync usage)

### Established Patterns
- Env-var-driven configuration: `Settings` dataclass with `os.getenv()` defaults — add `A2A_CLIENT_TIMEOUT` and per-agent `A2A_CLIENT_TIMEOUT_<NAME>` fields following the existing pattern
- Graceful degrade: every model/tool failure returns a fallback result, never raises — `@degrade_on_unavailable` follows the same philosophy for A2A
- `total=False` on all TypedDict fields — all degrade diffs must respect this (return only the keys that exist)
- Atomic file writes for logs — unchanged (debate_node logging stays local)

### Integration Points
- `graph.py` imports: remove 9 direct agent imports; replace with `from .a2a_client import ...`
- `graph.py` node functions: each node body changes from `agent_fn(state)` to `a2a_client.agent_fn(state)`; return value (diff dict) identical
- `debate_node`: loop body changes from `real_advocate({...})` / `fake_advocate({...})` to `a2a_client.real_advocate({...})` / `a2a_client.fake_advocate({...})`; catches `AgentUnavailableError` per-advocate for partial debate (D-05)
- `route_nei_check`: unchanged — if evidence agents return empty lists, NEI gate routes to judge regardless of whether the emptiness was from degradation or actual search results
- `config.py`: add `a2a_client_timeout` and per-agent timeout fields to `Settings`
</code_context>

<specifics>
## Specific Ideas

- The `@degrade_on_unavailable` decorator should log a warning at each call site so the operator can see which agents are down in the console output — consistent with existing `[Search] ...` message pattern.
- `AgentUnavailableError` should carry the agent name and the underlying cause (`str(exc)`) for debugging.
- The debate loop's partial-debate logic: on the first `AgentUnavailableError` from an advocate, mark that side as down and skip its turns for all remaining rounds. The available advocate still gets full turns each round.
- `evaluate_agent` wrapper (port 9002): exists for API completeness; marked with a docstring note that no graph node calls it (MCP uses it in-process directly).
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 4-LangGraph → A2A Client Wiring*
*Context gathered: 2026-08-16*
