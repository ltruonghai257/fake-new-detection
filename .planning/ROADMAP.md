# Roadmap: factcheck_agents v3.1 A2A Protocol Integration

**Milestone:** v3.1 · **Status:** Active

## Overview

Refactor all 10 factcheck agents to conform to the Google Agent2Agent (A2A) protocol (`a2a-sdk`).
Each agent becomes an independent HTTP service with a standard `TaskHandler` interface and an
Agent Card. LangGraph is retained as the routing orchestrator but its nodes call agents via
`A2AClient` HTTP requests instead of direct Python imports. The demo app's SSE bridge is updated
to route through A2A. The CLI, Python API, and MCP server remain externally unchanged.

## Phases

-   [x] **Phase 3: A2A Agent Wrappers & Launch Scripts** - Add `TaskHandler` to all 10 agents; serve each on its own uvicorn port; expose Agent Cards; write start/stop scripts
-   [~] **Phase 4: LangGraph → A2A Client Wiring** - Implement `a2a_client.py`; refactor `graph.py` nodes to call A2A HTTP instead of local functions; graceful-degrade for unreachable agents
-   [ ] **Phase 5: Demo App + Tests** - Update FastAPI SSE bridge to call A2A endpoints; update/add agent HTTP tests; update graph integration tests; verify backward compat

## Phase Details

### Phase 3: A2A Agent Wrappers & Launch Scripts

**Goal**: Wrap all 10 agents as A2A `TaskHandler` HTTP services and provide developer tooling to
start/stop the full agent fleet locally.
**Requirements**: A2A-01, A2A-02, A2A-03, A2A-03b
**Depends on**: Nothing (first v3.1 phase; v3.0 code stable)
**Success Criteria** (what must be TRUE):

1. `pip install "a2a-sdk[http-server,fastapi]"` installs without conflicts; version pinned in `factcheck_agents/` dependencies
2. Each of the 10 agent modules exposes a `TaskHandler` class with `async def process(task: Task) -> TaskResult`; `Task.input` carries the same fields the LangGraph state previously supplied
3. `GET /.well-known/agent.json` on each agent's port returns a valid A2A Agent Card JSON with `name`, `description`, `version`, `skills`, and `url`; validated against the A2A schema
4. `scripts/start_agents.sh` starts all 10 uvicorn processes on ports 9001–9010 in under 15 s on developer hardware; each port responds to `GET /.well-known/agent.json` within 5 s of startup
5. `scripts/stop_agents.sh` sends `SIGTERM` to all 10 processes and exits 0; no zombie processes; PID file cleaned up

Plans:

-   [x] 03-01: Wave 1 — Install `a2a-sdk[http-server,fastapi]` and add to dependencies; create `factcheck_agents/a2a_server.py` base module with shared uvicorn factory and Agent Card builder; define A2A port constants in `config.py` (A2A*PORT*\* env vars, defaults 9001–9010)
-   [x] 03-02: Wave 2 — Add `TaskHandler` + uvicorn app to evidence-side agents: `search_agent`, `evaluate_agent`, `real_source_agent`, `fake_source_agent`; each handler deserializes `Task.input` → existing agent function input types and serializes the return value to `TaskResult`
-   [x] 03-03: Wave 3 — Add `TaskHandler` + uvicorn app to pipeline agents: `social_loop_agent`, `agreement_gate`, `real_advocate` (split from `debate_node`), `fake_advocate` (split from `debate_node`), `judge_agent`, `conclusion_agent`
-   [x] 03-04: Wave 4 — Write `scripts/start_agents.sh` and `scripts/stop_agents.sh`; smoke-test script that curls all 10 `/.well-known/agent.json` endpoints and asserts HTTP 200

---

### Phase 4: LangGraph → A2A Client Wiring

**Goal**: Replace direct agent function calls inside LangGraph nodes with `A2AClient` HTTP calls;
handle agent unavailability gracefully.
**Requirements**: A2A-04, A2A-05, A2A-05b
**Depends on**: Phase 3 (all 10 A2A servers stable and testable)
**Success Criteria** (what must be TRUE):

1. `factcheck_agents/a2a_client.py` provides 10 typed wrapper functions (one per agent); each constructs a valid A2A `Task`, calls the agent HTTP endpoint via `httpx.AsyncClient`, and returns the deserialized Python type; connection errors → `AgentUnavailableError`
2. `graph.py` node functions contain no direct imports of agent functions; every agent invocation goes through `a2a_client.*`; `build_graph()` and `build_debate_graph()` signatures unchanged
3. `AgentUnavailableError` in any node triggers the same graceful-degrade behaviour as a missing checkpoint in v3.0 (pipeline continues; affected field marked `unavailable`)
4. Running `python -m factcheck_agents.cli "Tin tức test"` with all 10 agent servers running produces a verdict within 60 s; running it with all servers down produces `unavailable` gracefully (no crash, no unhandled exception)
5. Existing 83 unit tests (non-HTTP) still pass unchanged

Plans:

- [x] 04-01-PLAN.md
- [ ] 04-02-PLAN.md

1/2 plans executed

-   [~] 04-02: Wave 2 — Refactor `graph.py` node functions to call `a2a_client.*`; add `AgentUnavailableError` handlers in each node; update test patch targets; run existing unit tests to verify no regressions

---

### Phase 5: Demo App + Tests

**Goal**: Update the demo app SSE bridge to call A2A endpoints; add HTTP-level agent tests;
update graph integration tests; confirm the CLI/API/MCP interfaces are unchanged.
**Requirements**: A2A-06, A2A-06b, A2A-07, A2A-07b, A2A-08
**Depends on**: Phase 4 (stable A2A client + refactored graph)
**Success Criteria** (what must be TRUE):

1. `demo_app/backend/streaming.py` calls `a2a_client.*` for agent invocations; the SSE event schema (`stage_start`, `turn_start`, `chunk`, `turn_end`, `verdict`, `heartbeat`) is byte-for-byte identical to v3.0 output from the frontend's perspective; React frontend requires zero changes
2. If an A2A agent is unreachable during a demo session, the SSE stream emits `{"event":"stage_error","data":{"message":"..."}}` in Vietnamese and closes with HTTP 200 (no 500)
3. Each of the 10 `TaskHandler`s has a pytest test that spins up the uvicorn server in-process, sends a real A2A `Task`, and asserts the `TaskResult` fields match expected types
4. Graph integration tests (2 sample Vietnamese claims) run end-to-end with agent servers started as a session-scoped pytest fixture; full test suite passes in < 60 s on developer hardware
5. `factcheck_agents/cli.py`, `run_fact_check()`, and `mcp_server.py` pass their existing tests without modification; no breaking change detectable by callers

Plans:

-   [ ] 05-01: Wave 1 — Update `demo_app/backend/streaming.py` to call A2A agents; add `stage_error` SSE event type; manual smoke test confirms debate streaming still works end-to-end in the browser
-   [ ] 05-02: Wave 2 — Write 10 agent HTTP tests (one per `TaskHandler`); update graph integration tests with session-scoped agent-server fixture; run full suite and confirm < 60 s; verify CLI + MCP backward compat with existing callers

---

## Dependency Map

```
Phase 3 (A2A Wrappers) ──► Phase 4 (LangGraph Wiring) ──► Phase 5 (Demo + Tests)
```

Phase 3 must be complete and all 10 agent servers stable before Phase 4 begins.
Phase 4 must be complete and the A2A client stable before Phase 5 begins.

## Progress

| Phase | Name                   | Plans | Status   |
| ----- | ---------------------- | ----- | -------- |
| 3     | A2A Agent Wrappers     | 4/4   | Complete |
| 4     | LangGraph → A2A Wiring | 2/2   | In Progress|
| 5     | Demo App + Tests       | 2/2   | Pending  |

_Created: 2026-08-13_
