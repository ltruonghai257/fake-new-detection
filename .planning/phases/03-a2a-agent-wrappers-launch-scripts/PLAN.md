# Phase 3 Plan: A2A Agent Wrappers & Launch Scripts

**Phase:** 3 · **Plans:** 03-01 through 03-04 · **Waves:** 4 sequential
**Status:** Planning complete
**Requirements:** A2A-01, A2A-02, A2A-03, A2A-03b
**Decisions:** D-01 through D-15 (see CONTEXT.md)

---

## Plan 03-01: Install a2a-sdk + Create Base Server Module

**Goal:** Add `a2a-sdk[http-server,fastapi]` to project dependencies. Create shared `a2a_server.py` base module with uvicorn factory, AgentCardConfig dataclass, BaseTaskHandler abstract class, and serialization helpers. Add A2A port constants to `config.py`.

**Depends on:** Nothing (first plan in phase)

### Steps

#### 1. Add a2a-sdk dependency
- **File:** `pyproject.toml` (lines ~59-67, `agents` extra)
- Add `"a2a-sdk[http-server,fastapi]>=0.1.0"` to `[project.optional-dependencies] agents` list
- Run: `uv sync --extra agents` to install
- **Verify:** `uv pip list | grep a2a-sdk` shows installed version

#### 2. Add A2A port constants to config.py
- **File:** `factcheck_agents/config.py` (after line ~126, `checkpoint_db` field)
- Add 10 port fields to `Settings` dataclass, each reading from env var with default:
  - `A2A_PORT_SEARCH` (default 9001), `A2A_PORT_EVALUATE` (9002), `A2A_PORT_REAL_SOURCE` (9003), `A2A_PORT_FAKE_SOURCE` (9004), `A2A_PORT_SOCIAL_LOOP` (9005), `A2A_PORT_AGREEMENT_GATE` (9006), `A2A_PORT_REAL_ADVOCATE` (9007), `A2A_PORT_FAKE_ADVOCATE` (9008), `A2A_PORT_JUDGE` (9009), `A2A_PORT_CONCLUSION` (9010)
- Add convenience helper `def a2a_ports() -> dict[str, int]` that returns `{"search": 9001, ...}` mapping agent name to port
- **Verify:** `python -c "from factcheck_agents.config import settings; print(settings.A2A_PORT_SEARCH)"` → `9001`

#### 3. Create AgentCardConfig dataclass in a2a_server.py
- **File:** NEW `factcheck_agents/a2a_server.py`
- `AgentCardConfig` dataclass with fields: `name: str`, `description: str`, `version: str`, `skills: list[dict]`, `port: int`
- `build_agent_card(cfg: AgentCardConfig) -> dict`: returns dict matching A2A Agent Card JSON schema with `url` = `http://localhost:{port}`
- **Verify:** `python -c "from factcheck_agents.a2a_server import AgentCardConfig, build_agent_card; c = AgentCardConfig('test', 'desc', '1.0', [{'id': 'test', 'name': 'Test'}], 9001); print(build_agent_card(c)['url'])"` → `http://localhost:9001`

#### 4. Create serialization helpers in a2a_server.py
- **File:** `factcheck_agents/a2a_server.py` (continue)
- `serialize_state(state: FactCheckState) -> dict`: deep-copy + JSON-safe conversion of state dict for `Task.input` (handle datetime, Path, non-serializable types)
- `deserialize_state(data: dict) -> FactCheckState`: reverse — reconstruct typed dict from JSON-safe dict
- Use `orjson` (already in deps) for fast serialization
- **Verify:** `python -c "from factcheck_agents.a2a_server import serialize_state, deserialize_state; from factcheck_agents.state import FactCheckState; s = FactCheckState(statement='test'); d = serialize_state(s); s2 = deserialize_state(d); assert s2['statement'] == 'test'"`

#### 5. Create BaseTaskHandler abstract class in a2a_server.py
- **File:** `factcheck_agents/a2a_server.py` (continue)
- `BaseTaskHandler(TaskHandler)` abstract class inheriting from `a2a_sdk.server.tasks.TaskHandler`
- Abstract method: `async def agent_fn(self, state: FactCheckState) -> dict` — each agent implements this
- Concrete `async def process(self, task: Task) -> TaskResult`:
  1. Deserialize `task.input` → `FactCheckState` via `deserialize_state()`
  2. Call `self.agent_fn(state)` (await if async)
  3. Serialize returned dict → `TaskResult.output`
  4. On exception: return `TaskResult(status='failed', output={'error': str(e)})`
- `create_app(handler: BaseTaskHandler, cfg: AgentCardConfig) -> FastAPI`: factory that mounts handler routes + `/.well-known/agent.json` endpoint
- `run_server(handler: BaseTaskHandler, cfg: AgentCardConfig) -> None`: calls `uvicorn.run(create_app(handler, cfg), host='127.0.0.1', port=cfg.port, log_level='info')`
- **Verify:** No runtime errors on import: `python -c "from factcheck_agents.a2a_server import BaseTaskHandler, AgentCardConfig, create_app, run_server"`

#### 6. Update __init__.py exports
- **File:** `factcheck_agents/__init__.py`
- Add exports for `a2a_server` module (or keep internal — agents import directly)

---

## Plan 03-02: Add TaskHandler to Evidence-Side Agents

**Goal:** Wrap `search_agent`, `evaluate_agent`, `real_source_agent`, `fake_source_agent` as A2A TaskHandlers with uvicorn server entry points. Each agent file gets a `TaskHandler` class + `__main__` block for standalone execution.

**Depends on:** 03-01 (a2a_server.py, config ports)

### Steps

#### 1. Wrap search_agent
- **File:** `factcheck_agents/agents/search_agent.py` (append)
- Create `SearchAgentHandler(BaseTaskHandler)`:
  - `agent_fn(state)`: call existing `search_agent(state)`, return diff
  - `AGENT_CARD = AgentCardConfig(name='search_agent', description='Drafts search queries and retrieves web evidence', version='1.0', skills=[{'id': 'web_search', 'name': 'Web Search'}], port=settings.A2A_PORT_SEARCH)`
- Add `if __name__ == '__main__': run_server(SearchAgentHandler(), SearchAgentHandler.AGENT_CARD)` at bottom
- **Verify:** `python -m factcheck_agents.agents.search_agent` starts uvicorn on port 9001 (Ctrl+C to stop)

#### 2. Wrap evaluate_agent
- **File:** `factcheck_agents/agents/evaluate_agent.py` (append)
- Same pattern: `EvaluateAgentHandler(BaseTaskHandler)` wrapping `evaluate_agent(state)`
- AgentCardConfig: name='evaluate_agent', description='Runs PhoBERT and COOLANT models on statement', port=9002
- **Verify:** `python -m factcheck_agents.agents.evaluate_agent` starts on port 9002

#### 3. Wrap real_source_agent
- **File:** `factcheck_agents/agents/real_source_agent.py` (append)
- Same pattern: `RealSourceAgentHandler(BaseTaskHandler)` wrapping `real_source_agent(state)`
- AgentCardConfig: name='real_source_agent', description='Searches trusted Vietnamese news domains', port=9003
- **Verify:** `python -m factcheck_agents.agents.real_source_agent` starts on port 9003

#### 4. Wrap fake_source_agent
- **File:** `factcheck_agents/agents/fake_source_agent.py` (append)
- Same pattern: `FakeSourceAgentHandler(BaseTaskHandler)` wrapping `fake_source_agent(state)`
- AgentCardConfig: name='fake_source_agent', description='Searches flagged/non-official sources and Google Fact Check API', port=9004
- **Verify:** `python -m factcheck_agents.agents.fake_source_agent` starts on port 9004

#### 5. Wave 2 integration test
- Test: start all 4 agents, curl each `/.well-known/agent.json`, assert 200 + valid JSON with `name`, `description`, `version`, `skills`, `url`
- Test: send a minimal `Task` with `{"statement": "Hà Nội là thủ đô Việt Nam"}` to each agent, assert `TaskResult` with `status='completed'` and expected output keys
- **Verify:** All 4 agents respond correctly

---

## Plan 03-03: Add TaskHandler to Pipeline Agents + Split debate_node

**Goal:** Wrap `social_loop_agent`, `agreement_gate`, `real_advocate` (new), `fake_advocate` (new), `judge_agent`, `conclusion_agent`. Split `debate_node.py` per D-07/D-08. Update `agents/__init__.py`.

**Depends on:** 03-02 (a2a_server.py stable, evidence agents working)

### Steps

#### 1. Extract shared debate utilities
- **File:** NEW `factcheck_agents/agents/debate_utils.py`
- Move from `debate_node.py`: `_format_evidence()`, `_format_model_results()`, `_format_model_results_verdict()`, `_format_history()`, `_parse_advocate_json()`, `_append_turn()`
- Move `REAL_ADVOCATE_PROMPT` and `FAKE_ADVOCATE_PROMPT` (the prompt constants)
- **Verify:** `python -c "from factcheck_agents.agents.debate_utils import REAL_ADVOCATE_PROMPT, FAKE_ADVOCATE_PROMPT"` works

#### 2. Create real_advocate.py
- **File:** NEW `factcheck_agents/agents/real_advocate.py`
- `real_advocate(state: FactCheckState) -> dict`: single-turn LLM call using `REAL_ADVOCATE_PROMPT` from `debate_utils`. Reads `debate_turns` history from state, formats it, invokes LLM, returns `{"debate_turn": {...}, "messages": [...]}` for a single turn. Does NOT loop — the LangGraph edge handles iteration.
- `RealAdvocateHandler(BaseTaskHandler)`: wraps `real_advocate(state)`. `agent_fn` reads `debate_role='real'` from state meta to select correct prompt.
- AgentCardConfig: name='real_advocate', description='Defends claim as REAL in adversarial debate', port=9007
- **Verify:** `python -m factcheck_agents.agents.real_advocate` starts on port 9007

#### 3. Create fake_advocate.py
- **File:** NEW `factcheck_agents/agents/fake_advocate.py`
- `fake_advocate(state: FactCheckState) -> dict`: single-turn LLM call using `FAKE_ADVOCATE_PROMPT`. Same structure as real_advocate but defends FAKE.
- `FakeAdvocateHandler(BaseTaskHandler)`: wraps `fake_advocate(state)`. Reads `debate_role='fake'` from state meta.
- AgentCardConfig: name='fake_advocate', description='Defends claim as FAKE in adversarial debate', port=9008
- **Verify:** `python -m factcheck_agents.agents.fake_advocate` starts on port 9008

#### 4. Delete debate_node.py
- **File:** DELETE `factcheck_agents/agents/debate_node.py`
- Remove from `agents/__init__.py` exports
- Add `real_advocate` and `fake_advocate` to `agents/__init__.py`
- **Verify:** `python -c "from factcheck_agents.agents import real_advocate, fake_advocate"` works; `from factcheck_agents.agents import debate_node` raises ImportError

#### 5. Wrap social_loop_agent
- **File:** `factcheck_agents/agents/social_loop_agent.py` (append)
- Same pattern: `SocialLoopAgentHandler(BaseTaskHandler)` wrapping `social_loop_agent(state)`
- AgentCardConfig: name='social_loop_agent', description='Searches tiktok.com + flagged domains when evidence is weak', port=9005
- **Verify:** `python -m factcheck_agents.agents.social_loop_agent` starts on port 9005

#### 6. Wrap agreement_gate
- **File:** `factcheck_agents/agents/agreement_gate.py` (append)
- Same pattern: `AgreementGateHandler(BaseTaskHandler)` wrapping `agreement_gate(state)`
- AgentCardConfig: name='agreement_gate', description='Computes weighted agreement score; decides whether to skip debate', port=9006
- **Verify:** `python -m factcheck_agents.agents.agreement_gate` starts on port 9006

#### 7. Wrap judge_agent
- **File:** `factcheck_agents/agents/judge_agent.py` (append)
- Same pattern: `JudgeAgentHandler(BaseTaskHandler)` wrapping `judge_agent(state)`
- AgentCardConfig: name='judge_agent', description='Scores debate turns on 1-5 dimensions; computes weight breakdown', port=9009
- **Verify:** `python -m factcheck_agents.agents.judge_agent` starts on port 9009

#### 8. Wrap conclusion_agent
- **File:** `factcheck_agents/agents/conclusion_agent.py` (append)
- Same pattern: `ConclusionAgentHandler(BaseTaskHandler)` wrapping `conclusion_agent(state)`
- AgentCardConfig: name='conclusion_agent', description='Fuses model verdicts + evidence into final 4-class decision', port=9010
- **Verify:** `python -m factcheck_agents.agents.conclusion_agent` starts on port 9010

#### 9. Wave 3 integration test
- Test: start all 10 agents (via sequential script or manual), curl each `/.well-known/agent.json`, assert 200
- Test: send minimal `Task` to each of the 6 pipeline agents, assert valid `TaskResult`
- Test: send `Task` with `{"debate_role": "real"}` to port 9007, verify handler uses correct prompt
- Test: send `Task` with `{"debate_role": "fake"}` to port 9008, verify handler uses correct prompt
- **Verify:** All 10 agents respond correctly to A2A protocol

---

## Plan 03-04: Launch Scripts + Smoke Tests

**Goal:** Write `scripts/start_agents.sh` and `scripts/stop_agents.sh` for local dev. Create smoke test that validates all 10 agent endpoints.

**Depends on:** 03-03 (all 10 agents have working `__main__` entry points)

### Steps

#### 1. Create scripts directory + start_agents.sh
- **File:** NEW `scripts/start_agents.sh`
- Sequential startup: for each agent in order (search→evaluate→real_source→fake_source→social_loop→agreement_gate→real_advocate→fake_advocate→judge→conclusion):
  - Check port is free via `lsof -ti :<port>` — fail immediately if in use (D-11)
  - Start with `python -m factcheck_agents.agents.<name> > logs/agent_<name>.log 2>&1 &` (D-10)
  - Write PID to `.pids/<name>.pid` (D-13)
  - Short sleep between starts (0.5s) to avoid race conditions
- After all 10 started: poll each `http://localhost:<port>/.well-known/agent.json` in loop (max 30s timeout, 0.5s interval) (D-12)
- Print summary: `✓ All 10 agents ready (X.Xs)`
- **Verify:** `bash scripts/start_agents.sh` starts all 10 agents, output shows all ready

#### 2. Create stop_agents.sh
- **File:** NEW `scripts/stop_agents.sh`
- For each `.pid` file in `.pids/` directory:
  - Read PID
  - `kill -TERM <pid>` (graceful shutdown)
  - Wait up to 5s for process to exit
  - `kill -9 <pid>` as fallback if still alive
  - Remove `.pid` file
- Print summary: `✓ Stopped N agents`
- Exit 0 even if some PIDs already gone (clean shutdown) — but print warnings
- **Verify:** After `start_agents.sh`, run `stop_agents.sh` — all processes stopped, no zombie, `.pids/` empty

#### 3. Create logs + .pids directories
- **File:** NEW `logs/.gitkeep` (empty — ensures directory exists in repo)
- **File:** NEW `.pids/.gitkeep` (empty)
- Add `logs/` and `.pids/` to `.gitignore` (except `.gitkeep` files)
- **Verify:** `bash scripts/start_agents.sh` creates `logs/agent_*.log` files

#### 4. Create smoke test script
- **File:** NEW `scripts/smoke_test_agents.sh`
- For each of 10 ports (9001–9010):
  - `curl -s http://localhost:<port>/.well-known/agent.json | python -c "import sys,json; d=json.load(sys.stdin); assert 'name' in d; assert 'url' in d; print(f'✓ {d[\"name\"]} OK')"`
  - Assert exit code 0 per agent
- Print summary: `✓ N/10 agents responding`
- **Verify:** `bash scripts/smoke_test_agents.sh` passes with all 10 agents running

#### 5. End-to-end smoke test
- Start all 10 agents with `start_agents.sh`
- Run `smoke_test_agents.sh` — assert all 10 pass
- Stop all 10 with `stop_agents.sh`
- Verify no lingering processes: `lsof -ti :9001,:9002,...,:9010` returns empty
- **Verify:** Full start→smoke→stop cycle completes in < 15s (per success criterion #4)

---

## Verification Summary

| # | Criterion | Plan | Verify Method |
|---|-----------|------|---------------|
| 1 | a2a-sdk installs without conflicts; version pinned | 03-01.1 | `uv pip list \| grep a2a-sdk` |
| 2 | All 10 agents expose TaskHandler with `process(task) → TaskResult` | 03-02, 03-03 | Smoke test sends Task to each port |
| 3 | `/.well-known/agent.json` returns valid Agent Card per agent | 03-02, 03-03 | curl + JSON schema validation |
| 4 | `start_agents.sh` starts all 10 on ports 9001–9010 in < 15s | 03-04.1, 03-04.5 | Timed run |
| 5 | `stop_agents.sh` sends SIGTERM, cleans PIDs, no zombies | 03-04.2, 03-04.5 | Process check after stop |

## Files Changed

### New Files
- `factcheck_agents/a2a_server.py` — BaseTaskHandler, AgentCardConfig, serialization, uvicorn factory
- `factcheck_agents/agents/debate_utils.py` — Shared advocate utilities extracted from debate_node
- `factcheck_agents/agents/real_advocate.py` — Real advocate agent + TaskHandler
- `factcheck_agents/agents/fake_advocate.py` — Fake advocate agent + TaskHandler
- `scripts/start_agents.sh` — Sequential startup script
- `scripts/stop_agents.sh` — Graceful shutdown script
- `scripts/smoke_test_agents.sh` — Endpoint validation script
- `logs/.gitkeep` — Logs directory marker
- `.pids/.gitkeep` — PID directory marker

### Modified Files
- `pyproject.toml` — Add `a2a-sdk[http-server,fastapi]` to agents extra
- `factcheck_agents/config.py` — Add 10 A2A port constants + `a2a_ports()` helper
- `factcheck_agents/__init__.py` — Add a2a_server exports
- `factcheck_agents/agents/__init__.py` — Replace debate_node with real_advocate, fake_advocate
- `factcheck_agents/agents/search_agent.py` — Append SearchAgentHandler + __main__
- `factcheck_agents/agents/evaluate_agent.py` — Append EvaluateAgentHandler + __main__
- `factcheck_agents/agents/real_source_agent.py` — Append RealSourceAgentHandler + __main__
- `factcheck_agents/agents/fake_source_agent.py` — Append FakeSourceAgentHandler + __main__
- `factcheck_agents/agents/social_loop_agent.py` — Append SocialLoopAgentHandler + __main__
- `factcheck_agents/agents/agreement_gate.py` — Append AgreementGateHandler + __main__
- `factcheck_agents/agents/judge_agent.py` — Append JudgeAgentHandler + __main__
- `factcheck_agents/agents/conclusion_agent.py` — Append ConclusionAgentHandler + __main__
- `.gitignore` — Add `logs/` and `.pids/` entries

### Deleted Files
- `factcheck_agents/agents/debate_node.py` — Split into real_advocate + fake_advocate + debate_utils

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `a2a-sdk` API surface changes between versions | Breaking | Pin to exact version; smoke test catches regressions |
| debate_node.py consumers break (graph.py imports it) | Build failure | Update graph.py imports in this plan; Phase 4 handles A2A client wiring |
| Port conflicts in CI/dev environments | Start fails | `start_agents.sh` detects port-in-use and fails immediately with clear message |
| uvicorn startup race — agent responds before fully initialized | Flaky smoke test | Poll `/.well-known/agent.json` with timeout; don't trust `process started` output |
| `orjson` can't serialize datetime/Path in FactCheckState | Task input corruption | `serialize_state()` handles these types explicitly; smoke test catches |

---

*Generated: 2026-08-15*
