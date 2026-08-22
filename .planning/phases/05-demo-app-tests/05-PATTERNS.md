# Phase 5 — Pattern Map

## 1. SSE streaming.py — _post() and done patterns

**File:** `demo_app/backend/streaming.py`

### Function signature and setup (lines 103–128)
```python
async def sse_stream(
    request_id: str,
    statement: str,
    image_path: str | None,
    use_phobert: bool = True,
    use_coolant: bool = True,
    use_evidence: bool = True,
) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def _post(evt: dict) -> None:
        """Thread-safe event posting (D-02)."""
        loop.call_soon_threadsafe(queue.put_nowait, evt)
```

### run_graph() — the exact inner loop structure (lines 129–315)
**Insert point for stage_error check is immediately after `accumulated.update(node_output)` (line 157), before the `stage = NODE_STAGE_MAP.get(node_name)` block:**
```python
    def run_graph() -> None:
        try:
            from factcheck_agents.graph import build_debate_graph, initial_state

            graph = build_debate_graph(checkpointer=None)
            state = initial_state(statement, image_path, language="vi")
            # ... state field assignments ...
            accumulated: dict = {}
            emitted_stages: set[str] = set()
            stream_config = {"configurable": {"thread_id": request_id}}

            for chunk in graph.stream(state, config=stream_config):
                if done.is_set():
                    break
                node_name, node_output = next(iter(chunk.items()))
                if node_output is None:
                    continue
                accumulated.update(node_output)

                # ← STAGE_ERROR CHECK INSERTS HERE (after accumulated.update, before stage_start)

                # Emit stage_start for node transitions (D-09, D-10)
                stage = NODE_STAGE_MAP.get(node_name)
                if stage and stage not in emitted_stages:
                    emitted_stages.add(stage)
                    _post({"type": "stage_start", "name": stage})

                # Emit stage_log ...
                # Debate turn re-chunking with inner done.is_set() checks ...

        except Exception as exc:
            _post({"type": "error", "error": str(exc)})
        finally:
            _post({"type": "_done"})
```

### done.set() / done.is_set() usage — all existing sites
```python
# In for loop at top:
if done.is_set():
    break

# In debate turn inner loop (line 179, 193):
if done.is_set():
    break

# In _heartbeat():
while not done.is_set():
    ...
    if not done.is_set():
        queue.put_nowait({"type": "heartbeat"})

# In finally block (lines 338–339) — always called on exit:
finally:
    done.set()
    hb_task.cancel()
    executor.shutdown(wait=False)
```

### stage_error insertion pattern (from CONTEXT/RESEARCH decisions)
```python
# After accumulated.update(node_output), before stage = NODE_STAGE_MAP.get(node_name):
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

### Existing except block (lines 311–314) — do NOT modify
```python
        except Exception as exc:
            _post({"type": "error", "error": str(exc)})
        finally:
            _post({"type": "_done"})
```

---

## 2. TaskHandler — agreement_gate example

**File:** `factcheck_agents/agents/agreement_gate.py` (lines 140–166)

```python
# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server


class AgreementGateHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`agreement_gate` over HTTP (port 9006)."""

    agent_card_config = AgentCardConfig(
        name="agreement_gate",
        description="Computes weighted agreement score; decides whether to skip debate",
        version="1.0",
        skills=[
            {
                "id": "agreement",
                "name": "Agreement Scoring",
                "description": "Weighted model+evidence agreement computation",
            }
        ],
        port=settings.a2a_port_agreement_gate,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return agreement_gate(state)


if __name__ == "__main__":
    run_server(AgreementGateHandler(), AgreementGateHandler.agent_card_config)
```

**Key structural notes:**
- `agent_card_config` is a **class attribute** (not instance), typed as `AgentCardConfig`
- `agent_fn` is `async def` but wraps a synchronous function — `BaseTaskHandler._run_agent_sync` handles the sync/async bridge via `asyncio.to_thread`
- `run_server(handler_instance, handler_class.agent_card_config)` at module level under `if __name__ == "__main__":`
- Port sourced from `settings.a2a_port_<name>` attribute (not hardcoded)

---

## 3. create_app() / BaseTaskHandler interface

**File:** `factcheck_agents/a2a_server.py`

### create_app() signature (lines 242–272)
```python
def create_app(handler: BaseTaskHandler, cfg: AgentCardConfig) -> FastAPI:
    """Build the FastAPI app with A2A JSON-RPC + REST + agent-card routes."""
    agent_card = build_agent_card(cfg)
    request_handler = DefaultRequestHandler(
        agent_executor=handler,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI(title=cfg.name, version=cfg.version)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/"),
        rest_routes=create_rest_routes(request_handler),
    )
    # /.well-known/agent.json alias inserted at route index 0
    return app
```

### run_server() signature (lines 275–281)
```python
def run_server(handler: BaseTaskHandler, cfg: AgentCardConfig) -> None:
    """Run the agent service with uvicorn on ``cfg.port`` (blocks)."""
    import uvicorn
    uvicorn.run(
        create_app(handler, cfg), host="127.0.0.1", port=cfg.port, log_level="info"
    )
```

### BaseTaskHandler interface (lines 144–236)
```python
class BaseTaskHandler(AgentExecutor):
    agent_card_config: AgentCardConfig  # class attribute, set by subclass

    async def agent_fn(self, state: FactCheckState) -> dict:
        """Run the agent logic and return the state diff. May be sync or async."""
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Handles full task lifecycle: deserialize -> run -> artifact -> complete/fail
        ...

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        ...
```

### AgentCardConfig dataclass (lines 64–72)
```python
@dataclass
class AgentCardConfig:
    name: str
    description: str
    version: str
    skills: List[dict] = field(default_factory=list)
    port: int = 9001
```

### Test fixture usage pattern
```python
from factcheck_agents.a2a_server import create_app
from factcheck_agents.agents.agreement_gate import AgreementGateHandler

app = create_app(AgreementGateHandler(), AgreementGateHandler.agent_card_config)
# → usable with uvicorn.Server(uvicorn.Config(app, ...)) or httpx.Client
```

---

## 4. Test file patterns

### test_a2a_client.py — import style, fixture usage, assertion patterns
**File:** `tests/factcheck_agents/test_a2a_client.py`

```python
# Import style — named imports from production modules, no conftest fixtures needed
from unittest.mock import patch

import httpx
import pytest

from factcheck_agents import a2a_client
from factcheck_agents.a2a_client import (
    AgentUnavailableError,
    _timeout_for,
    agreement_gate,
    ...
)
from factcheck_agents.config import Settings, a2a_ports, settings
```

```python
# Mocking pattern — patch at the usage site (factcheck_agents.a2a_client.httpx.Client)
def test_call_agent_success_path(monkeypatch):
    monkeypatch.setattr(settings, "a2a_client_timeout", None)
    monkeypatch.setattr(settings, "a2a_client_timeout_search", 120)
    port = a2a_ports()["search_agent"]
    inner = {"evidence": [{"url": "https://a.com"}], "search_queries": ["q"]}
    response = {
        "task": {
            "status": {"state": "TASK_STATE_COMPLETED"},
            "artifacts": [{"name": "output", "parts": [{"data": inner}]}],
        }
    }
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.return_value.json.return_value = response
        result = a2a_client.call_agent("search_agent", {"statement": "s"})

    assert result == inner
    mock_client.assert_called_once_with(timeout=120.0)
    args, kwargs = mock_client.return_value.post.call_args
    assert args[0] == f"http://localhost:{port}/message:send"
    assert kwargs["headers"] == {"A2A-Version": "1.0"}
    body = kwargs["json"]["message"]
    assert body["role"] == "ROLE_USER"
    assert body["messageId"].startswith("msg-")
    assert isinstance(body["parts"][0]["data"], dict)
```

```python
# pytest.raises pattern
def test_call_agent_http_error_raises_unavailable():
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.side_effect = httpx.ConnectError("conn refused")
        with pytest.raises(AgentUnavailableError) as excinfo:
            a2a_client.call_agent("search_agent", {"statement": "s"})
    assert excinfo.value.agent_name == "search_agent"
    assert "conn refused" in excinfo.value.cause
```

### test_debate_pipeline_integration.py — fixture shape, mock state, @pytest.mark usage
**File:** `tests/factcheck_agents/test_debate_pipeline_integration.py`

```python
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import pytest
from langgraph.checkpoint.memory import MemorySaver
from factcheck_agents.graph import build_debate_graph, initial_state
```

```python
# autouse function-scope fixture (no session scope here)
@pytest.fixture(autouse=True)
def setup_logs_dir():
    """Create logs directory before test session to avoid first-run mkdir errors."""
    Path("logs").mkdir(exist_ok=True)
```

```python
# patch.multiple at graph import namespace — mocks all agent node functions at once
with patch.multiple(
    "factcheck_agents.graph",
    real_source_agent=lambda s: {"evidence_real": [...], "messages": []},
    fake_source_agent=lambda s: {"evidence_fake": [...], "messages": []},
    reranker_node=lambda s: {"evidence": [...], "consistency_score": 0.8},
    verify_agent=lambda s: {"model_results": [...], "messages": []},
    judge_agent=lambda s: {"verdict": _make_verdict(), "weight_breakdown": {...}, "messages": []},
):
    graph = build_debate_graph(checkpointer=MemorySaver())
    result = graph.invoke(
        initial_state(claim), config={"configurable": {"thread_id": "t-wc"}}
    )
    assert isinstance(result["verdict"], dict)
    assert result["verdict"]["label"] in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIED", "NEI"}
```

```python
# Mock state shape helpers used across tests:
def _fake_evidence(domain: str) -> dict:
    return {
        "title": "T",
        "url": f"https://{domain}/a",
        "snippet": "snippet",
        "source_tier": "trusted",
        "score": 0.9,
    }

def _make_verdict(label="TRUE") -> dict:
    return {
        "label": label,
        "verdict_binary": "REAL",
        "verdict_label_vi": "Thật",
        "confidence": 0.85,
        "rationale": "test",
        "citations": [],
    }
```

**Note:** No `@pytest.mark.integration` in existing tests — all run by default. New Phase 5 tests add this marker for opt-in gating.

---

## 5. thread_id pattern — cli.py

**File:** `factcheck_agents/cli.py` (lines 67–71)

```python
# cli.py uses hash-based thread_id (Phase 4 fix):
graph = build_graph()
result = graph.invoke(
    initial_state(args.statement, image_path=args.image, language=args.language),
    config={"configurable": {"thread_id": f"cli-{hash(args.statement)}"}},
)
```

**Phase 5 uses `uuid.uuid4()` instead of hash (per D-10):**
```python
import uuid  # add at top of file

# In run_fact_check() / mcp_server.py fact_check():
result = graph.invoke(
    state,
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
)
```

**Note:** `uuid` is already imported in `a2a_client.py` (`import uuid` at line 7) — same import style to use.

---

## 6. A2A client call pattern

**File:** `factcheck_agents/a2a_client.py` (lines 62–110)

### call_agent() function signature
```python
def call_agent(agent_name: str, state: FactCheckState) -> dict:
    """Send the serialized FactCheckState to an A2A agent and return its diff."""
```

### Request body shape (lines 70–86)
```python
port = a2a_ports()[agent_name]
timeout = _timeout_for(agent_name)
payload = serialize_state(dict(state))
request_body = {
    "message": {
        "role": "ROLE_USER",
        "parts": [{"data": payload}],
        "messageId": f"msg-{uuid.uuid4()}",
    }
}
resp = httpx.Client(timeout=timeout).post(
    f"http://localhost:{port}/message:send",
    json=request_body,
    headers={"A2A-Version": "1.0"},
)
resp.raise_for_status()
```

### Response parsing (lines 91–110)
```python
data = resp.json()
# SDK 1.1.x REST wraps the Task in a "task" key: {"task": {...}}
task = data.get("task") or data
task_state = task.get("status", {}).get("state", "")
artifacts = task.get("artifacts", [])

if task_state == "TASK_STATE_FAILED":
    error_data = _extract_artifact_data(artifacts)
    raise AgentUnavailableError(agent_name, port, error_data.get("error", "server error"))
if task_state == "TASK_STATE_WORKING":
    return {}
# TASK_STATE_COMPLETED (or unknown) — extract the diff
return _extract_artifact_data(artifacts)
```

### _extract_artifact_data helper (lines 52–59)
```python
def _extract_artifact_data(artifacts: list) -> dict:
    for artifact in artifacts:
        if artifact.get("name") == "output":
            parts = artifact.get("parts", [])
            if parts and "data" in parts[0]:
                return parts[0]["data"]
    return {}
```

### TaskResult assertion pattern (from test_a2a_client.py)
```python
response = {
    "task": {
        "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [{"name": "output", "parts": [{"data": <diff_dict>}]}],
    }
}
# Assert:
assert result == diff_dict
assert kwargs["headers"] == {"A2A-Version": "1.0"}
assert body["role"] == "ROLE_USER"
assert body["messageId"].startswith("msg-")
```

---

## 7. a2a_ports() mapping

**File:** `factcheck_agents/config.py` (lines 224–237)

```python
def a2a_ports() -> dict[str, int]:
    """Map agent name → A2A server port (contract with start_agents.sh / Phase 4)."""
    return {
        "search_agent":      settings.a2a_port_search,      # default 9001
        "evaluate_agent":    settings.a2a_port_evaluate,    # default 9002
        "real_source_agent": settings.a2a_port_real_source, # default 9003
        "fake_source_agent": settings.a2a_port_fake_source, # default 9004
        "social_loop_agent": settings.a2a_port_social_loop, # default 9005
        "agreement_gate":    settings.a2a_port_agreement_gate, # default 9006
        "real_advocate":     settings.a2a_port_real_advocate,  # default 9007
        "fake_advocate":     settings.a2a_port_fake_advocate,  # default 9008
        "judge_agent":       settings.a2a_port_judge,       # default 9009
        "conclusion_agent":  settings.a2a_port_conclusion,  # default 9010
    }
```

**Settings fields for each port** (env var → default):
```
A2A_PORT_SEARCH          → 9001   (settings.a2a_port_search)
A2A_PORT_EVALUATE        → 9002   (settings.a2a_port_evaluate)
A2A_PORT_REAL_SOURCE     → 9003   (settings.a2a_port_real_source)
A2A_PORT_FAKE_SOURCE     → 9004   (settings.a2a_port_fake_source)
A2A_PORT_SOCIAL_LOOP     → 9005   (settings.a2a_port_social_loop)
A2A_PORT_AGREEMENT_GATE  → 9006   (settings.a2a_port_agreement_gate)
A2A_PORT_REAL_ADVOCATE   → 9007   (settings.a2a_port_real_advocate)
A2A_PORT_FAKE_ADVOCATE   → 9008   (settings.a2a_port_fake_advocate)
A2A_PORT_JUDGE           → 9009   (settings.a2a_port_judge)
A2A_PORT_CONCLUSION      → 9010   (settings.a2a_port_conclusion)
```

**Test usage pattern:**
```python
from factcheck_agents.config import a2a_ports
port = a2a_ports()["agreement_gate"]  # → 9006
```

---

## 8. run_fact_check() current state

**File:** `factcheck_agents/__init__.py` (lines 28–39)

```python
def run_fact_check(
    statement: str, image_path: str | None = None, language: str = "auto"
):
    """Convenience one-shot entrypoint. Builds the graph and runs it once."""
    from .graph import build_graph, initial_state

    graph = build_graph()
    state = initial_state(statement, image_path=image_path, language=language)
    result = graph.invoke(state)                             # ← BUG: missing thread_id config
    result["verdict_binary"] = result.get("verdict", {}).get("verdict_binary")
    result["verdict_label_vi"] = result.get("verdict", {}).get("verdict_label_vi")
    return result
```

**Fix target:** `graph.invoke(state)` on line 36 — add `config={"configurable": {"thread_id": str(uuid.uuid4())}}`.

**Also add** `import uuid` at top of file (not currently imported).

**External API:** `run_fact_check(statement, image_path, language)` signature and return dict are unchanged.

---

## 9. mcp_server.py current invoke

**File:** `factcheck_agents/mcp_server.py` (lines 39–56)

```python
@mcp.tool()
def fact_check(
    statement: str, image_path: Optional[str] = None, language: str = "auto"
) -> dict:
    """Run the full Search -> Evaluate -> Conclusion pipeline on a claim."""
    graph = build_graph()
    result = graph.invoke(
        initial_state(statement, image_path=image_path, language=language)
    )                                                        # ← BUG: missing thread_id config
    return {
        "statement": statement,
        "verdict": result.get("verdict", {}),
        "model_results": result.get("model_results", []),
        "evidence": result.get("evidence", []),
        "search_queries": result.get("search_queries", []),
        "verdict_binary": result.get("verdict", {}).get("verdict_binary"),
        "verdict_label_vi": result.get("verdict", {}).get("verdict_label_vi"),
    }
```

**Fix target:** `graph.invoke(initial_state(...))` on lines 45–47 — add `config=` kwarg:
```python
    result = graph.invoke(
        initial_state(statement, image_path=image_path, language=language),
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
```

**Also add** `import uuid` at top of file (not currently imported; current imports are `from __future__ import annotations`, `from typing import Optional`, `from mcp.server.fastmcp import FastMCP`, etc.).

---

## PATTERN MAPPING COMPLETE
