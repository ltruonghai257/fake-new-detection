# Phase 3: A2A Agent Wrappers & Launch Scripts - Context

**Gathered:** 2026-08-15
**Status:** Ready for planning

## Phase Boundary

Wrap all 10 existing factcheck agents as A2A `TaskHandler` HTTP services (uvicorn on ports 9001–9010), expose standard Agent Cards (`/.well-known/agent.json`), and provide `start_agents.sh` / `stop_agents.sh` developer tooling. This phase does NOT wire LangGraph nodes to call the A2A services — that's Phase 4.

## Implementation Decisions

### Task Input Contract
- **D-01:** `Task.input` carries the full serialized `FactCheckState` dict — every handler deserializes the complete state and passes it directly to the existing agent function signature. Minimizes Phase 4 migration cost. — **Reversibility:** costly — switching to per-agent minimal schemas later would touch all 10 handlers and the Phase 4 A2A client.
- **D-02:** `TaskResult.output` contains only the state diff — the keys the agent mutated (same shape as LangGraph node return today). Caller merges the diff into its local state.
- **D-03:** Serialization/deserialization helpers live in `a2a_server.py` (shared base module). Single place to update if `FactCheckState` fields change.
- **D-04:** On exception during processing: `TaskResult.status = 'failed'`, `TaskResult.output = {'error': str(e)}`. Caller detects `failed` status and triggers the `AgentUnavailableError` graceful-degrade path.
- **D-05:** Advocate role identity is encoded in `Task.input` as a `debate_role: 'real' | 'fake'` key. The handler reads it and selects the correct prompt. Both real and fake advocate services (ports 9007/9008) use the same handler code, differentiated by this runtime field.
- **D-06:** Full debate conversation history is passed in `Task.input` each call (stateless handler). Growing payload is acceptable — `max_debate_rounds` is capped at 2, so total history stays small.

### debate_node Split
- **D-07:** `debate_node.py` is split into `agents/real_advocate.py` and `agents/fake_advocate.py`. Each exports its own `TaskHandler` class. The original `debate_node.py` is deleted.
- **D-08:** Shared debate utilities (prompt templates, LLM invocation helpers, JSON parsing) are extracted into a new `agents/debate_utils.py`. Both advocate files import from it.

### Launch Script Behavior
- **D-09:** `start_agents.sh` starts agents sequentially (one at a time). Slower startup but easy to debug individual failures.
- **D-10:** Uvicorn output goes to per-agent log files in `logs/agent_<name>.log`. Clean separation for troubleshooting 10 concurrent agents.
- **D-11:** Port-already-in-use is a hard failure — script stops immediately with an error message.
- **D-12:** After starting all 10 agents, the script blocks until every port responds to `GET /.well-known/agent.json` (poll loop with timeout). Returns success only when all agents are confirmed ready.
- **D-13:** PID tracking uses per-agent `.pid` files in a `.pids/` directory (e.g., `.pids/search_agent.pid`). `stop_agents.sh` reads each `.pid` file, sends `SIGTERM`, and cleans up.

### Base Server Module Scope
- **D-14:** `a2a_server.py` provides a full `BaseTaskHandler` abstract class with a shared `process()` flow: deserialize `FactCheckState` from `Task.input` → call `self.agent_fn(state)` → serialize returned diff to `TaskResult.output` → return. Each agent file implements only `agent_fn`. — **Reversibility:** one-way — the abstract base class becomes a contract all 10 handlers depend on; changing the flow requires updating every handler.
- **D-15:** Agent Card configuration uses a per-agent `AgentCardConfig` dataclass exported from `a2a_server.py`. Each agent file creates its own instance with name, description, skills, and version. Shared format, per-agent values.

### Claude's Discretion
- Task input uses full `FactCheckState` (not per-agent minimal fields) — simplifies migration.
- Serialization lives in shared `a2a_server.py` (not per-agent inline).
- Uvicorn logs go to per-agent log files.
- Base module includes a `BaseTaskHandler` abstract class (not just uvicorn factory + cards).
- Agent Card content uses a shared `AgentCardConfig` dataclass with per-agent instances.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Decisions & Scope
- `.planning/PROJECT.md` — v3.1 milestone definition, target features, out-of-scope list, stack decisions
- `.planning/REQUIREMENTS.md` — A2A-01 through A2A-03b requirement details with traceability
- `.planning/ROADMAP.md` — Phase 3 success criteria, plan breakdown (03-01 through 03-04), dependency map
- `.planning/STATE.md` — Port assignments (9001–9010), tunable defaults (agreement threshold, weights, max_rounds)

### Existing Code (MUST read before implementing wrappers)
- `factcheck_agents/config.py` — `Settings` dataclass, env-var patterns (add A2A port constants here)
- `factcheck_agents/state.py` — `FactCheckState` TypedDict (all fields that `Task.input` must serialize)
- `factcheck_agents/agents/search_agent.py` — `search_agent(state)` function signature and return type
- `factcheck_agents/agents/evaluate_agent.py` — `evaluate_agent(state)` function signature and return type
- `factcheck_agents/agents/real_source_agent.py` — function signature
- `factcheck_agents/agents/fake_source_agent.py` — function signature
- `factcheck_agents/agents/social_loop_agent.py` — function signature
- `factcheck_agents/agents/agreement_gate.py` — function signature
- `factcheck_agents/agents/debate_node.py` — current combined advocate logic (will be split per D-07, D-08)
- `factcheck_agents/agents/judge_agent.py` — function signature
- `factcheck_agents/agents/conclusion_agent.py` — function signature

### External Specs
- Google A2A Protocol (`a2a-sdk[http-server,fastapi]`) — `Task`, `TaskResult`, `TaskHandler`, Agent Card schema. No project-local A2A spec doc exists; refer to the SDK documentation at install time.

## Existing Code Insights

### Reusable Assets
- `factcheck_agents/config.py` `Settings` dataclass: established pattern for env-var-driven configuration. Add `A2A_PORT_*` constants here with defaults 9001–9010.
- `factcheck_agents/state.py` `FactCheckState`: the TypedDict used by every agent. Reuse directly — no new intermediate types needed.
- `factcheck_agents/helpers.py`: existing async HTTP helpers that A2A handlers may need for LLM calls.

### Established Patterns
- Agent functions take `state: FactCheckState` and return `dict` (the mutated keys). This pattern is directly mapped to the `BaseTaskHandler.process()` flow.
- All settings come from environment variables (no hardcoded secrets). A2A port constants follow the same pattern.
- Graceful degradation: unavailable models produce `unavailable` results, never crash. A2A error handling (D-04) mirrors this — `failed` status triggers the same degrade path.

### Integration Points
- `factcheck_agents/config.py` — add `A2A_PORT_*` env vars here (one per agent, defaults 9001–9010)
- `factcheck_agents/agents/__init__.py` — update exports when splitting `debate_node.py` and adding new files
- Phase 4 (`a2a_client.py`) will consume these agent HTTP endpoints — the Agent Card schema and port constants are the contract between phases

## Specific Ideas

No specific requirements beyond what's captured in decisions — open to standard approaches.

## Deferred Ideas

None — discussion stayed within phase scope.

---

*Phase: 3-A2A Agent Wrappers & Launch Scripts*
*Context gathered: 2026-08-15*
