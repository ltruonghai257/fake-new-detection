# Phase 5: Demo App + Tests — Research

## Summary

Phase 5 requires updating the SSE bridge to detect agent unavailability and emit stage_error events, writing 10 in-process HTTP tests for all agent TaskHandlers, adding graph integration tests with real agent servers, and fixing thread_id crashes in run_fact_check() and mcp_server.py. The SSE bridge already calls A2A endpoints via the graph (Phase 4 completed), so only the stage_error detection logic is needed. All 10 agents follow a consistent TaskHandler pattern with create_app() factory for test instantiation. The thread_id fix follows the exact pattern already implemented in cli.py.

---

## 1. SSE stage_error Implementation (A2A-06b)

**Location:** `demo_app/backend/streaming.py`

**Function:** `run_graph()` (lines 129–314)

**Insert point:** After `accumulated.update(node_output)` (line ~157) and before the stage_start emission — inside the `for chunk in graph.stream(...)` loop:

```
for chunk in graph.stream(state, config=stream_config):
    if done.is_set():
        break
    node_name, node_output = next(iter(chunk.items()))
    if node_output is None:
        continue
    accumulated.update(node_output)
    # ← INSERT STAGE_ERROR CHECK HERE
```

**`_post()` call signature:**
- Function at line ~125: `_post(evt: dict)`
- Implementation: `loop.call_soon_threadsafe(queue.put_nowait, evt)`
- Usage: `_post({"type": "stage_error", "data": {"message": "..."}})`

**`done` object:**
- Type: `asyncio.Event()` (line ~123)
- `done.set()` signals shutdown; `done.is_set()` checks status
- Current loop checks at lines ~152, 179, 193: `if done.is_set(): break`
- Finally block (line ~339): calls `done.set()` on cleanup

**Scan logic (from CONTEXT.md D-01):**
```python
messages = node_output.get("messages", [])
if any(
    "unavailable" in str(part)
    for msg in messages
    for part in (msg if isinstance(msg, (list, tuple)) else [msg])
):
    _post({"type": "stage_error", "data": {"message": "Một số dịch vụ tạm thời không khả dụng. Không thể hoàn thành phân tích."}})
    done.set()
    break
```

**Vietnamese message (fixed, per D-02):**
`"Một số dịch vụ tạm thời không khả dụng. Không thể hoàn thành phân tích."`

**Existing error handling (do NOT modify):**
- Lines ~311–314: `except Exception as exc` → `_post({"type": "error", ...})` → `_post({"type": "_done"})`
- stage_error is a separate detection path inside the loop, not a replacement for this

---

## 2. Agent HTTP Tests (A2A-07)

**TaskHandler interface:**
- Base class: `BaseTaskHandler` in `a2a_server.py` (lines ~144–237)
- Required method: `async def agent_fn(self, state: FactCheckState) -> dict`
- Required attribute: `agent_card_config: AgentCardConfig`
- Return: state diff dict (only keys the agent mutates)

**`create_app()` usage:**
- Signature: `create_app(handler: BaseTaskHandler, cfg: AgentCardConfig) -> FastAPI`
- Returns a FastAPI app usable directly with uvicorn or httpx.Client
- Example:
  ```python
  from factcheck_agents.a2a_server import create_app
  from factcheck_agents.agents.search_agent import SearchAgentHandler
  app = create_app(SearchAgentHandler(), SearchAgentHandler.agent_card_config)
  ```

**A2A Task JSON payload shape (from `a2a_client.py` lines ~73–80):**
```
POST /message:send
Headers: {"A2A-Version": "1.0"}
Body: {
    "message": {
        "role": "ROLE_USER",
        "parts": [{"data": <serialized FactCheckState dict>}],
        "messageId": "msg-<uuid>"
    }
}
```

**Response / TaskResult assertion:**
- Response body: `{"task": {...}}`
- Assert `task.status.state == "TASK_STATE_COMPLETED"`
- Output in `task.artifacts[0].parts[0].data`
- Assert diff contains expected keys per agent

**`a2a_ports()` mapping (`config.py` lines ~224–237):**
```
search_agent:      9001
evaluate_agent:    9002
real_source_agent: 9003
fake_source_agent: 9004
social_loop_agent: 9005
agreement_gate:    9006
real_advocate:     9007
fake_advocate:     9008
judge_agent:       9009
conclusion_agent:  9010
```

**All 10 agents — handler, returns, and env requirements:**

| # | Agent | Handler class | Returns (diff keys) | Env vars |
|---|-------|--------------|---------------------|----------|
| 1 | search_agent (9001) | `SearchAgentHandler` | evidence, search_queries, claim_variants, evidence_graph, messages | TAVILY_API_KEY or GOOGLE_CSE_API_KEY+ID |
| 2 | evaluate_agent (9002) | `EvaluateAgentHandler` | model_results, messages | checkpoint paths only |
| 3 | real_source_agent (9003) | `RealSourceAgentHandler` | evidence_real, messages, evidence_workflow_steps | TAVILY_API_KEY or GOOGLE_CSE_API_KEY |
| 4 | fake_source_agent (9004) | `FakeSourceAgentHandler` | evidence_fake, messages | TAVILY_API_KEY, optional GOOGLE_FACTCHECK_API_KEY |
| 5 | social_loop_agent (9005) | `SocialLoopAgentHandler` | evidence_social, social_loop_fired, messages, errors | TAVILY_API_KEY |
| 6 | agreement_gate (9006) | `AgreementGateHandler` | agreement_score, weight_breakdown, debate_exit_reason | None (pure calculation) |
| 7 | real_advocate (9007) | `RealAdvocateHandler` | debate_turn, messages | OPENAI_API_KEY |
| 8 | fake_advocate (9008) | `FakeAdvocateHandler` | debate_turn, messages | OPENAI_API_KEY |
| 9 | judge_agent (9009) | `JudgeAgentHandler` | weight_breakdown, messages | OPENAI_API_KEY |
| 10 | conclusion_agent (9010) | `ConclusionAgentHandler` | verdict, messages | OPENAI_API_KEY |

**Import paths:**
```python
from factcheck_agents.agents.search_agent import SearchAgentHandler
from factcheck_agents.agents.evaluate_agent import EvaluateAgentHandler
from factcheck_agents.agents.real_source_agent import RealSourceAgentHandler
from factcheck_agents.agents.fake_source_agent import FakeSourceAgentHandler
from factcheck_agents.agents.social_loop_agent import SocialLoopAgentHandler
from factcheck_agents.agents.agreement_gate import AgreementGateHandler
from factcheck_agents.agents.real_advocate import RealAdvocateHandler
from factcheck_agents.agents.fake_advocate import FakeAdvocateHandler
from factcheck_agents.agents.judge_agent import JudgeAgentHandler
from factcheck_agents.agents.conclusion_agent import ConclusionAgentHandler
```

**Port-conflict skip pattern (D-04):**
```python
import socket
s = socket.socket()
result = s.connect_ex(("127.0.0.1", port))
s.close()
if result == 0:
    pytest.skip(f"Port {port} already in use — skip to avoid conflict with running agent server")
```

---

## 3. Graph Integration Tests (A2A-07b)

**uvicorn-in-thread fixture pattern (D-07):**
```python
import uvicorn, threading, time, httpx

config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
# Poll readiness
deadline = time.time() + 10
while time.time() < deadline:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/.well-known/agent.json", timeout=1)
        if r.status_code == 200:
            break
    except Exception:
        time.sleep(0.1)
```

**8 agents to start (per D-07):**
```
real_source_agent:  9003
fake_source_agent:  9004
social_loop_agent:  9005
agreement_gate:     9006
real_advocate:      9007
fake_advocate:      9008
judge_agent:        9009
conclusion_agent:   9010
```
Excluded: search_agent (M1 path only), evaluate_agent (unused by both graphs)

**Existing test file NOT to modify:**
- `tests/factcheck_agents/test_debate_pipeline_integration.py` — mocked integration tests, per D-06

**New file:** `tests/factcheck_agents/test_a2a_integration.py`
- Session-scoped fixture starts all 8 agents
- Tests marked `@pytest.mark.integration`
- Fixture defined in-file (no conftest.py needed — consistent with existing pattern)

**Minimal claim examples (D-08, < 30s each on developer hardware):**
- `"Hà Nội là thủ đô của Việt Nam"` (factually true, fast evidence)
- `"Mặt trăng làm từ phô mai"` (Moon is made of cheese — easily debunked, short debate)

**`run_fact_check()` signature and return type:**
- Location: `factcheck_agents/__init__.py` lines ~28–39
- Signature: `def run_fact_check(statement: str, image_path: str | None = None, language: str = "auto") -> dict`
- Returns: full graph state dict with keys `verdict`, `verdict_binary`, `verdict_label_vi`, etc.

---

## 4. thread_id Fix (A2A-08)

**`factcheck_agents/__init__.py` — exact location:**
- Line ~36: `result = graph.invoke(state)` ← fix target
- `uuid` not currently imported
- Fix:
  ```python
  import uuid  # add at top
  # ...
  result = graph.invoke(state, config={"configurable": {"thread_id": str(uuid.uuid4())}})
  ```

**`factcheck_agents/mcp_server.py` — exact location:**
- Line ~45: `result = graph.invoke(initial_state(...))` ← fix target
- `uuid` not currently imported
- Fix:
  ```python
  import uuid  # add at top
  # ...
  result = graph.invoke(
      initial_state(statement, image_path=image_path, language=language),
      config={"configurable": {"thread_id": str(uuid.uuid4())}}
  )
  ```

**Pattern to copy — `cli.py` (Phase 4 fix):**
```python
config={"configurable": {"thread_id": f"cli-{hash(args.statement)}"}}
```
Phase 5 uses `str(uuid.uuid4())` instead of hash (per D-10 — uuid for true uniqueness)

**External API:** No signature changes. `run_fact_check()` callers see zero diff.

---

## 5. Validation Architecture

**A2A-06b (SSE stage_error):**
- Manual: Start demo app, stop one A2A agent, submit claim → SSE stream must emit `stage_error` event and close with HTTP 200 (no 500)
- Automated: `pytest tests/demo_app/test_streaming.py` — mock `@degrade_on_unavailable` to return unavailable diff, assert stage_error emitted

**A2A-07 (Agent HTTP tests):**
- Command: `pytest tests/factcheck_agents/test_agent_http.py -m integration -v`
- Requires: OPENAI_API_KEY, TAVILY_API_KEY (or GOOGLE_CSE_API_KEY) in environment
- Default run (no -m flag) skips all integration tests automatically

**A2A-07b (Graph integration tests):**
- Command: `pytest tests/factcheck_agents/test_a2a_integration.py -m integration -v`
- Session fixture starts 8 agents; total < 60s on developer hardware
- Requires same env vars as A2A-07

**A2A-08 (thread_id fix):**
- Automated: `pytest tests/factcheck_agents/ -k "entrypoint"` — patch graph.invoke, assert config with thread_id is passed
- Backward compat: `python -m factcheck_agents.cli "test"` exits 0; no signature changes

**Full regression gate:**
- `pytest tests/ -m "not integration"` → 99+ tests pass (same as Phase 4 baseline)
- No new failures in existing test suite

---

## 6. Implementation Notes

**Gotchas:**

1. **SSE scan:** Messages arrive as `[("assistant", "text")]` tuples — scan both `part[0]` and `part[1]` via `isinstance(msg, (list, tuple))` to cover both formats.

2. **Agent HTTP tests:** Use real `httpx.Client` (not `starlette.testclient.TestClient`) to exercise the full HTTP layer including A2A-Version header handling.

3. **uvicorn daemon threads:** `daemon=True` ensures threads die when the pytest session exits — no orphan processes.

4. **evaluate_agent:** Needs model checkpoint files, not API keys. Tests should `pytest.skip` if checkpoint path env vars are missing.

5. **Port conflict guard:** Apply to both the 10 single-agent HTTP tests and the 8-agent session fixture. Skip, don't fail.

6. **`uuid` import:** Neither `__init__.py` nor `mcp_server.py` imports `uuid` currently — add `import uuid` to both.

**Ordering (independent work streams):**
- A2A-06b (SSE) → independent
- A2A-07 (agent HTTP tests) → independent
- A2A-07b (graph integration tests) → depends on A2A-07 agents working correctly
- A2A-08 (thread_id fix) → independent

**Files to create:**
- `tests/factcheck_agents/test_agent_http.py`
- `tests/factcheck_agents/test_a2a_integration.py`

**Files to modify:**
- `demo_app/backend/streaming.py`
- `factcheck_agents/__init__.py`
- `factcheck_agents/mcp_server.py`

**Files NOT to modify:**
- `tests/factcheck_agents/test_debate_pipeline_integration.py` (per D-06)
- `factcheck_agents/a2a_client.py`
- `factcheck_agents/a2a_server.py`
- React frontend (out of scope)

---

## RESEARCH COMPLETE
