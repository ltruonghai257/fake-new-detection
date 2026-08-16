# Phase 4: LangGraph → A2A Client Wiring — Research

**Researched:** 2026-08-16
**Status:** Complete — ready for planning

---

## Technical Research

### Q1: JSON-RPC `message.send` Request/Response Format

The server (`a2a_server.py`) registers both JSON-RPC (`POST /`) and REST routes via `add_a2a_routes_to_fastapi()`. The JSON-RPC endpoint at `POST /` handles the `message.send` method.

**Request format** (JSON-RPC 2.0 envelope):

```json
{
  "jsonrpc": "2.0",
  "method": "message.send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {
          "data": {
            "statement": "Tuyên bố kiểm tra",
            "evidence": [],
            "search_queries": [],
            "...": "..."
          }
        }
      ],
      "messageId": "msg-<uuid>"
    }
  },
  "id": "<uuid>"
}
```

The `params` field is a `SendMessageRequest` (from the A2A protobuf schema). The `message.parts[0].data` carries the serialized `FactCheckState` dict — exactly what `serialize_state()` produces. The `messageId` is optional but should be unique.

**Response format** (on success):

```json
{
  "jsonrpc": "2.0",
  "result": {
    "id": "<task-uuid>",
    "contextId": "<context-uuid>",
    "status": {
      "state": "TASK_STATE_COMPLETED",
      "message": {
        "role": "agent",
        "parts": [{"text": "Task completed successfully"}]
      }
    },
    "artifacts": [
      {
        "artifactId": "<uuid>",
        "name": "output",
        "parts": [
          {
            "data": {
              "evidence": [...],
              "search_queries": [...],
              "messages": [["assistant", "[Search] ..."]]
            }
          }
        ]
      }
    ]
  },
  "id": "<uuid>"
}
```

The state diff is in `result.artifacts[0].parts[0].data`. If the agent fails, `result.status.state` is `"TASK_STATE_FAILED"` and `artifacts[0].parts[0].data` contains `{"error": "..."}`.

**Key detail — task lifecycle**: The server's `BaseTaskHandler.execute()` is awaited within the `send_message` handler. Because `_run_agent()` uses `asyncio.to_thread()`, the agent body runs in a worker thread while the event loop remains responsive. However, `execute()` **blocks until the agent completes** before returning the response. This means the `SendMessageResponse` contains the terminal task state (COMPLETED or FAILED) — no polling needed for the synchronous case.

**Edge case**: If the SDK ever changes to fire-and-forget `execute()`, the response would contain a WORKING task and polling `GET /tasks/{id}` would be needed. We handle this by checking the task state in the response and polling if not terminal.

**Extracting the diff**: `response["result"]["artifacts"][0]["parts"][0]["data"]`. The `messages` field is serialized as a list of lists (e.g., `[["assistant", "..."]]`) because tuples become lists during `orjson` round-trip. This is safe — all downstream consumers iterate with `for role, content in messages`, which unpacks lists identically to tuples. No code checks `isinstance(msg, tuple)`.

---

### Q2: REST vs JSON-RPC — Which Is Simpler?

**REST endpoint**: `POST http://localhost:{port}/message:send` with the same `SendMessageRequest` body (no JSON-RPC envelope). Response is the raw `Task` object.

**JSON-RPC endpoint**: `POST http://localhost:{port}/` with the JSON-RPC envelope wrapping `SendMessageRequest`. Response is `SendMessageResponse` with the `result` wrapper.

**Decision: Use REST** (`POST /message:send`).

| Factor | REST | JSON-RPC |
|--------|------|----------|
| Request body | `SendMessageRequest` dict directly | `{"jsonrpc":"2.0", "method":"message.send", "params": {...}, "id":"..."}` |
| Response body | Task dict directly | `{"jsonrpc":"2.0", "result": <Task>, "id":"..."}` |
| Error handling | HTTP status codes | JSON-RPC error object inside 200 |
| Code complexity | 1 level of nesting | 2 levels (envelope + result) |
| SDK parity | Same handler underneath | Same handler underneath |

REST is simpler: one fewer nesting level in both request and response. The A2A SDK creates both routes from the same handler, so behavior is identical. JSON-RPC adds the `jsonrpc`/`method`/`id` envelope that provides no value for our local-only, sync-only use case.

**Implementation**: Use `httpx.Client.post(f"http://localhost:{port}/message:send", json=request_body)`.

---

### Q3: Sync Bridge Pattern

**Context**: Graph nodes are sync (LangGraph's `StateGraph` uses sync `.invoke()`). The A2A SDK client (`a2a.client.Client`) is async-only. All 9 graph nodes + 4 entry points (`run_fact_check`, `cli.py`, `mcp_server.py`, `streaming.py`) use sync `.invoke()`.

**D-01 Decision**: Use `httpx.Client` (sync) or `asyncio.run` shim — zero caller changes.

**Recommendation: `httpx.Client` (synchronous HTTP client)**.

```python
import httpx

def call_agent(port: int, state: FactCheckState, timeout: float) -> dict:
    """Send serialized state to agent at `port`, return deserialized diff."""
    payload = serialize_state(dict(state))
    request_body = {
        "message": {
            "role": "user",
            "parts": [{"data": payload}],
            "messageId": f"msg-{uuid.uuid4()}",
        }
    }
    try:
        response = httpx.Client(timeout=timeout).post(
            f"http://localhost:{port}/message:send",
            json=request_body,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AgentUnavailableError(port=port, cause=str(exc)) from exc

    task = response.json()
    state = task.get("status", {}).get("state", "")
    artifacts = task.get("artifacts", [])

    if state == "TASK_STATE_FAILED":
        error_data = _extract_artifact_data(artifacts)
        raise AgentUnavailableError(
            port=port,
            cause=error_data.get("error", "unknown server error"),
        )

    # TASK_STATE_COMPLETED — extract diff
    return _extract_artifact_data(artifacts)
```

**Why not `asyncio.run(httpx.AsyncClient...)`?**
- `asyncio.run()` creates a new event loop each call → overhead per agent invocation
- `httpx.Client` is simpler, fewer lines, no event-loop nesting issues
- No benefit to async client since we need sync output

**Why not `a2a.client.Client`?**
- Async-only; would need `asyncio.run()` wrapper
- Protobuf-based; adds dependency complexity
- No value over raw HTTP for our simple use case (serialize state → POST → deserialize diff)

**Connection reuse**: Create one `httpx.Client` per wrapper call (stateless). For 10 agents called ~15 times in a full pipeline run, connection overhead is negligible. If profiling shows it matters later, add a session-scoped client — but not now (YAGNI).

---

### Q4: Serialized State After `serialize_state()`

`serialize_state()` (line 131-133 of `a2a_server.py`) does:
```python
def serialize_state(state: FactCheckState) -> dict:
    return orjson.loads(orjson.dumps(_json_safe(dict(state))))
```

`_json_safe()` converts:
- `datetime`/`date` → ISO-8601 string
- `Path` → string
- `set`/`tuple` → list
- `dict` → recursive conversion (keys → str)
- **Non-JSON-serializable objects** (e.g., `EvidenceGraph`) → `repr(value)` string

**What `EvidenceGraph` becomes**: After CR-01 fix, `evidence_graph` in the serialized diff is `"<factcheck_agents.graph_utils.EvidenceGraph object at 0x...>"` — a repr string. The client wrapper MUST NOT pass this to downstream consumers. Instead, reconstruct locally:

```python
# In search_agent wrapper:
diff = call_agent(port, state, timeout)
evidence = diff.get("evidence", [])
if evidence:
    from factcheck_agents.graph_utils import EvidenceGraph
    diff["evidence_graph"] = EvidenceGraph.build_from_evidence(evidence)
else:
    diff["evidence_graph"] = None
```

This matches D-03 exactly.

**What `messages` becomes**: Tuples like `("assistant", "[Search] ...")` become lists `["assistant", "[Search] ..."]` after `orjson` round-trip. All downstream consumers unpack with `for role, content in messages` — works identically. No normalization needed.

**What `DebateTurn` becomes**: The debate turn dict is plain JSON (strings, ints, floats, lists). No special types. Round-trips cleanly.

---

### Q5: `@degrade_on_unavailable` Decorator

**Design**: A decorator applied to each wrapper function. On `AgentUnavailableError`, returns a degrade diff instead of propagating the exception.

```python
import functools
import logging

logger = logging.getLogger(__name__)

class AgentUnavailableError(Exception):
    """Raised when an A2A agent is unreachable or returns a failed task."""
    def __init__(self, agent_name: str, port: int, cause: str = ""):
        self.agent_name = agent_name
        self.port = port
        self.cause = cause
        super().__init__(f"[{agent_name}:{port}] unavailable: {cause}")

def degrade_on_unavailable(agent_name: str, degrade_diff: dict):
    """Decorator: catch AgentUnavailableError, log warning, return degrade_diff."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            try:
                return fn(state, *args, **kwargs)
            except AgentUnavailableError as exc:
                logger.warning(
                    "[%s] agent unavailable (port %s): %s — degraded",
                    agent_name, exc.port, exc.cause,
                )
                return dict(degrade_diff)  # shallow copy to avoid mutation
        return wrapper
    return decorator
```

**Per-agent degrade diffs** (defined in `a2a_client.py`):

| Agent | Degrade Diff | Notes |
|-------|-------------|-------|
| `search_agent` | `{"evidence": [], "search_queries": [], "claim_variants": [], "evidence_graph": None, "messages": [("assistant", "[Search] agent unavailable — degraded")]}` | Empty evidence → pipeline proceeds on model output only |
| `conclusion_agent` | `{"verdict": {"label": "UNVERIFIED", "verdict_binary": "NEI", "verdict_label_vi": "Chưa xác thực", "confidence": 0.0, "rationale": "Conclusion agent unavailable", "citations": [], "recommendation": ""}, "messages": [("assistant", "[Conclusion] agent unavailable — degraded")]}` | NEI verdict, 0 confidence |
| `real_source_agent` | `{"evidence_real": [], "messages": [("assistant", "[RealSource] agent unavailable — degraded")]}` | Empty list — `route_nei_check` handles this |
| `fake_source_agent` | `{"evidence_fake": [], "messages": [("assistant", "[FakeSource] agent unavailable — degraded")]}` | Empty list |
| `social_loop_agent` | `{"evidence_social": [], "social_loop_fired": True, "messages": [("assistant", "[SocialLoop] agent unavailable — degraded")]}` | `social_loop_fired=True` prevents re-entry (SOCLOOP-03) |
| `agreement_gate` | `{"agreement_score": 0.0, "weight_breakdown": {"phobert": 0.0, "coolant": 0.0, "evidence": 0.0}, "debate_exit_reason": ""}` | Score 0.0 → always routes to debate |
| `real_advocate` | `{"debate_turn": {"agent": "real", "error": "agent_unavailable"}, "messages": [("assistant", "[RealAdvocate] agent unavailable — degraded")]}` | D-05 partial debate |
| `fake_advocate` | `{"debate_turn": {"agent": "fake", "error": "agent_unavailable"}, "messages": [("assistant", "[FakeAdvocate] agent unavailable — degraded")]}` | D-05 partial debate |
| `judge_agent` | `{"weight_breakdown": {"phobert": 0.0, "coolant": 0.0, "evidence": 0.0, "debate": 0.0, "phobert_conf": 0.0, "coolant_conf": 0.0, "debate_conf": 0.0, "debate_direction": 0.0, "argument_scores": [], "debate_winner": "none", "model_signal": 0.0}, "messages": [("assistant", "[Judge] agent unavailable — degraded")]}` | All-zero breakdown → expert_agent produces NEI |
| `evaluate_agent` | `{"model_results": [], "messages": [("assistant", "[Evaluate] agent unavailable — degraded")]}` | No model results → agreement_gate returns 0.0 |

**Important**: All degrade diffs respect `total=False` on TypedDicts — they return only the keys that exist on `FactCheckState`.

---

### Q6: Timeout Env Vars for `config.py`

**D-06 specifies**:
- LLM agents (search, real_source, fake_source, real_advocate, fake_advocate, judge, conclusion) = 120s
- Calculation-only (agreement_gate) = 30s
- Web-search (social_loop) = 30s
- Model-inference (evaluate_agent) = 60s
- Global override: `A2A_CLIENT_TIMEOUT`
- Per-agent override: `A2A_CLIENT_TIMEOUT_<NAME>`

**Add to `Settings` dataclass in `config.py`**:

```python
# ── A2A client timeouts (Phase 4) ───────────────────────────────────────
a2a_client_timeout: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT", "0"))
)
a2a_client_timeout_search: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_SEARCH", "0"))
)
a2a_client_timeout_conclusion: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_CONCLUSION", "0"))
)
a2a_client_timeout_real_source: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_REAL_SOURCE", "0"))
)
a2a_client_timeout_fake_source: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_FAKE_SOURCE", "0"))
)
a2a_client_timeout_social_loop: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_SOCIAL_LOOP", "0"))
)
a2a_client_timeout_agreement_gate: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_AGREEMENT_GATE", "0"))
)
a2a_client_timeout_real_advocate: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_REAL_ADVOCATE", "0"))
)
a2a_client_timeout_fake_advocate: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_FAKE_ADVOCATE", "0"))
)
a2a_client_timeout_judge: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_JUDGE", "0"))
)
a2a_client_timeout_evaluate: float = field(
    default_factory=lambda: float(os.getenv("A2A_CLIENT_TIMEOUT_EVALUATE", "0"))
)
```

**Resolution logic in `a2a_client.py`**:

```python
_DEFAULT_TIMEOUTS = {
    "search_agent": 120,
    "conclusion_agent": 120,
    "real_source_agent": 120,
    "fake_source_agent": 120,
    "social_loop_agent": 30,
    "agreement_gate": 30,
    "real_advocate": 120,
    "fake_advocate": 120,
    "judge_agent": 120,
    "evaluate_agent": 60,
}

def _timeout_for(agent_name: str) -> float:
    """Resolve timeout: per-agent env > global env > hardcoded default."""
    from .config import settings
    per_agent = getattr(settings, f"a2a_client_timeout_{agent_name.split('_')[0]}", 0)
    # Per-agent name mapping: "search_agent" → "search", "real_source_agent" → "real_source", etc.
    # Actually simpler: map agent_name → Settings field suffix directly
    suffix = {
        "search_agent": "search",
        "conclusion_agent": "conclusion",
        "real_source_agent": "real_source",
        "fake_source_agent": "fake_source",
        "social_loop_agent": "social_loop",
        "agreement_gate": "agreement_gate",
        "real_advocate": "real_advocate",
        "fake_advocate": "fake_advocate",
        "judge_agent": "judge",
        "evaluate_agent": "evaluate",
    }.get(agent_name, agent_name)
    per_agent_val = getattr(settings, f"a2a_client_timeout_{suffix}", 0)
    if per_agent_val > 0:
        return per_agent_val
    if settings.a2a_client_timeout > 0:
        return settings.a2a_client_timeout
    return _DEFAULT_TIMEOUTS.get(agent_name, 30)
```

**D-07**: No automatic retry. `httpx.ConnectError` / `httpx.TimeoutException` → `AgentUnavailableError` immediately.

---

### Q7: `debate_node()` Refactor for A2A with Partial Debate

**Current code** (`graph.py:94-167`):
```python
def debate_node(state: FactCheckState) -> dict:
    # ... LLM guard ...
    for round_num in range(settings.max_debate_rounds):
        real_out = real_advocate({...state, "debate_role": "real"})
        real_turn = real_out["debate_turn"]
        turns.append(real_turn)
        # ... error check ...
        fake_out = fake_advocate({...state, "debate_role": "fake"})
        fake_turn = fake_out["debate_turn"]
        turns.append(fake_turn)
        # ... convergence check ...
```

**Refactored** (D-05 partial debate):

```python
def debate_node(state: FactCheckState) -> dict:
    from .a2a_client import real_advocate as a2a_real, fake_advocate as a2a_fake
    from .a2a_client import AgentUnavailableError

    # ... LLM guard (unchanged) ...

    real_down = False
    fake_down = False

    for round_num in range(settings.max_debate_rounds):
        # Real advocate turn
        if not real_down:
            try:
                real_out = a2a_real({**state, "debate_turns": turns, "debate_role": "real"})
            except AgentUnavailableError:
                real_down = True
                turns.append({"agent": "real", "error": "agent_unavailable"})
                logger.warning("[Debate] real_advocate unavailable — partial debate")
            else:
                real_turn = real_out.get("debate_turn")
                turns.append(real_turn)
                if real_turn is None or real_turn.get("error"):
                    exit_reason = "llm_error" if real_turn else "no_llm"
                    break
        else:
            turns.append({"agent": "real", "error": "agent_unavailable"})

        # Fake advocate turn
        if not fake_down:
            try:
                fake_out = a2a_fake({**state, "debate_turns": turns, "debate_role": "fake"})
            except AgentUnavailableError:
                fake_down = True
                turns.append({"agent": "fake", "error": "agent_unavailable"})
                logger.warning("[Debate] fake_advocate unavailable — partial debate")
            else:
                fake_turn = fake_out.get("debate_turn")
                turns.append(fake_turn)
                if fake_turn is None or fake_turn.get("error"):
                    exit_reason = "llm_error" if fake_turn else "no_llm"
                    break
        else:
            turns.append({"agent": "fake", "error": "agent_unavailable"})

        # Both down → exit
        if real_down and fake_down:
            exit_reason = "agent_unavailable"
            break

        # Skip convergence check if partial debate
        if real_down or fake_down:
            continue

        real_v = str(real_turn.get("verdict", "")).upper()
        fake_v = str(fake_turn.get("verdict", "")).upper()
        if real_v == fake_v and real_v in {"REAL", "FAKE"}:
            converged = True
            agreed_verdict = real_v
            exit_reason = "converged"
            break

    if not exit_reason:
        exit_reason = "max_rounds"

    # ... return (unchanged structure) ...
```

**Key D-05 semantics**:
- First `AgentUnavailableError` from an advocate → mark that side down, skip its turns for remaining rounds
- Available advocate still gets full turns each round
- `debate_turns` entries for unavailable side: `{"agent": "<side>", "error": "agent_unavailable"}`
- Both down → `debate_exit_reason = "agent_unavailable"`, judge runs without argument quality signal (confidence capped at 0.7 per JUDGE-02)
- One-sided debate (only one advocate available): convergence check is skipped (can't converge alone)

**Also fix IN-07**: Move `turns.append()` after None check (append only non-None turns from available advocates):

```python
real_turn = real_out.get("debate_turn")
if real_turn is not None:
    turns.append(real_turn)
```

---

## Implementation Patterns

### Pattern 1: Base `call_agent()` Function

```python
# factcheck_agents/a2a_client.py

import uuid
import logging
import httpx
from typing import Any
from .state import FactCheckState
from .a2a_server import serialize_state
from .config import settings, a2a_ports

logger = logging.getLogger(__name__)

class AgentUnavailableError(Exception):
    def __init__(self, agent_name: str = "", port: int = 0, cause: str = ""):
        self.agent_name = agent_name
        self.port = port
        self.cause = cause
        super().__init__(f"[{agent_name}:{port}] unavailable: {cause}")

_DEFAULT_TIMEOUTS = { ... }  # as defined in Q6

def _timeout_for(agent_name: str) -> float: ...

def call_agent(agent_name: str, state: FactCheckState) -> dict:
    """Send state to A2A agent, return deserialized diff."""
    port = a2a_ports()[agent_name]
    timeout = _timeout_for(agent_name)
    payload = serialize_state(dict(state))
    request_body = {
        "message": {
            "role": "user",
            "parts": [{"data": payload}],
            "messageId": f"msg-{uuid.uuid4()}",
        }
    }
    try:
        resp = httpx.Client(timeout=timeout).post(
            f"http://localhost:{port}/message:send",
            json=request_body,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise AgentUnavailableError(agent_name, port, str(exc)) from exc

    task = resp.json()
    task_state = task.get("status", {}).get("state", "")
    artifacts = task.get("artifacts", [])

    if task_state == "TASK_STATE_FAILED":
        error_data = _extract_artifact_data(artifacts)
        raise AgentUnavailableError(agent_name, port, error_data.get("error", "server error"))

    # TASK_STATE_COMPLETED
    return _extract_artifact_data(artifacts)

def _extract_artifact_data(artifacts: list) -> dict:
    """Extract data from the 'output' artifact."""
    for a in artifacts:
        if a.get("name") == "output":
            parts = a.get("parts", [])
            if parts and "data" in parts[0]:
                return parts[0]["data"]
    return {}
```

### Pattern 2: Typed Wrapper Function Template

```python
def search_agent(state: FactCheckState) -> dict:
    """Call search_agent via A2A (port 9001)."""
    diff = call_agent("search_agent", state)
    # D-03: rebuild EvidenceGraph locally
    evidence = diff.get("evidence", [])
    if evidence:
        from factcheck_agents.graph_utils import EvidenceGraph
        diff["evidence_graph"] = EvidenceGraph.build_from_evidence(evidence)
    else:
        diff["evidence_graph"] = None
    return diff

search_agent = degrade_on_unavailable("search_agent", _DEGRADE_SEARCH)(search_agent)
```

But cleaner: use the decorator syntax:

```python
@degrade_on_unavailable("search_agent", _DEGRADE_SEARCH)
def search_agent(state: FactCheckState) -> dict:
    diff = call_agent("search_agent", state)
    evidence = diff.get("evidence", [])
    if evidence:
        from factcheck_agents.graph_utils import EvidenceGraph
        diff["evidence_graph"] = EvidenceGraph.build_from_evidence(evidence)
    else:
        diff["evidence_graph"] = None
    return diff
```

### Pattern 3: Wrappers That Need No Post-Processing

Most wrappers are one-liners — just `call_agent` + degrade decorator:

```python
@degrade_on_unavailable("conclusion_agent", _DEGRADE_CONCLUSION)
def conclusion_agent(state: FactCheckState) -> dict:
    return call_agent("conclusion_agent", state)

@degrade_on_unavailable("real_source_agent", _DEGRADE_REAL_SOURCE)
def real_source_agent(state: FactCheckState) -> dict:
    return call_agent("real_source_agent", state)

@degrade_on_unavailable("fake_source_agent", _DEGRADE_FAKE_SOURCE)
def fake_source_agent(state: FactCheckState) -> dict:
    return call_agent("fake_source_agent", state)

@degrade_on_unavailable("social_loop_agent", _DEGRADE_SOCIAL_LOOP)
def social_loop_agent(state: FactCheckState) -> dict:
    return call_agent("social_loop_agent", state)

@degrade_on_unavailable("agreement_gate", _DEGRADE_AGREEMENT)
def agreement_gate(state: FactCheckState) -> dict:
    return call_agent("agreement_gate", state)

@degrade_on_unavailable("real_advocate", _DEGRADE_REAL_ADVOCATE)
def real_advocate(state: FactCheckState) -> dict:
    return call_agent("real_advocate", state)

@degrade_on_unavailable("fake_advocate", _DEGRADE_FAKE_ADVOCATE)
def fake_advocate(state: FactCheckState) -> dict:
    return call_agent("fake_advocate", state)

@degrade_on_unavailable("judge_agent", _DEGRADE_JUDGE)
def judge_agent(state: FactCheckState) -> dict:
    return call_agent("judge_agent", state)

@degrade_on_unavailable("evaluate_agent", _DEGRADE_EVALUATE)
def evaluate_agent(state: FactCheckState) -> dict:
    return call_agent("evaluate_agent", state)
```

Only `search_agent` needs post-processing (EvidenceGraph rebuild).

### Pattern 4: Graph Import Changes

**Before** (`graph.py`):
```python
from .agents import conclusion_agent, search_agent, social_search_agent, verify_agent
from .agents.agreement_gate import agreement_gate, route_after_agreement
from .agents.expert_agent import expert_agent
from .agents.fake_advocate import fake_advocate
from .agents.fake_source_agent import fake_source_agent
from .agents.judge_agent import judge_agent
from .agents.real_advocate import real_advocate
from .agents.real_source_agent import real_source_agent
from .agents.social_loop_agent import social_loop_agent
```

**After**:
```python
# Local agents (no A2A)
from .agents import social_search_agent, verify_agent
from .agents.expert_agent import expert_agent
from .agents.agreement_gate import route_after_agreement

# A2A client wrappers
from .a2a_client import (
    agreement_gate,
    conclusion_agent,
    fake_advocate,
    fake_source_agent,
    judge_agent,
    real_advocate,
    real_source_agent,
    search_agent,
    social_loop_agent,
)
```

**Note**: `agreement_gate` is imported from `a2a_client` (the wrapper), but `route_after_agreement` stays from `agents.agreement_gate` (it's a routing function, not an agent — never called via A2A).

### Pattern 5: Test Patch Target Update

**`test_graceful_degrade.py`** — change 4 `@patch` decorators:

```python
# Before:
@patch("factcheck_agents.graph.conclusion_agent")
@patch("factcheck_agents.graph.social_search_agent")
@patch("factcheck_agents.graph.verify_agent")
@patch("factcheck_agents.graph.search_agent")

# After:
@patch("factcheck_agents.a2a_client.conclusion_agent")
# social_search_agent stays local — no patch change needed
# verify_agent stays local — no patch change needed
@patch("factcheck_agents.a2a_client.search_agent")
```

Wait — `social_search_agent` and `verify_agent` are NOT A2A. They stay as direct imports. So only `search_agent` and `conclusion_agent` patches change target. But the test currently patches all 4 at `factcheck_agents.graph.*`. After refactor, `graph.py` imports `search_agent` and `conclusion_agent` from `a2a_client`, so patches must target `factcheck_agents.a2a_client.*`.

Updated patch structure:
```python
@patch("factcheck_agents.a2a_client.conclusion_agent")
@patch("factcheck_agents.graph.social_search_agent")    # unchanged — still local
@patch("factcheck_agents.graph.verify_agent")            # unchanged — still local
@patch("factcheck_agents.a2a_client.search_agent")
def test_graceful_degrade_no_crash(mock_search, mock_verify, mock_social, mock_concl):
```

**`test_debate_pipeline_integration.py`** — change `patch.multiple` target:

```python
# Before:
with patch.multiple(
    "factcheck_agents.graph",
    real_source_agent=...,
    fake_source_agent=...,
    reranker_node=...,
    verify_agent=...,
    judge_agent=...,
):

# After:
with patch.multiple(
    "factcheck_agents.a2a_client",
    real_source_agent=...,
    fake_source_agent=...,
    judge_agent=...,
):
with patch.multiple(
    "factcheck_agents.graph",
    reranker_node=...,    # unchanged — still local
    verify_agent=...,     # unchanged — still local
):
```

Two separate `patch.multiple` blocks — one for `a2a_client` (A2A wrappers), one for `graph` (local nodes). This is because `patch.multiple` only patches one namespace at a time.

---

## Risks & Unknowns

### Risk 1: `httpx.Client` Not Installed
- **Likelihood**: Low — httpx is a common dependency; the project likely already has it transitively (FastAPI, uvicorn, openai all depend on httpx).
- **Mitigation**: Check `pip list | grep httpx` before implementing. If missing, add to dependencies.

### Risk 2: REST Endpoint `POST /message:send` Returns Non-Terminal Task
- **Likelihood**: Low — the server's `execute()` blocks until completion. But the SDK might change behavior.
- **Mitigation**: Add a check: if `task_state` is `TASK_STATE_WORKING`, poll `GET /tasks/{id}` with exponential backoff (max 5 attempts, 0.5s→1s→2s→4s→8s). This is 3 lines of code and handles the edge case without adding complexity to the normal path.

### Risk 3: `messages` Tuple→List Conversion Breaking Some Consumer
- **Likelihood**: Very low — the codebase consistently uses `for role, content in messages` unpacking. No code checks `isinstance(msg, tuple)`.
- **Mitigation**: If needed, normalize in the client wrapper: `[(tuple(m) if isinstance(m, list) else m) for m in messages]`. But this is defensive coding — skip it unless tests fail.

### Risk 4: `agreement_gate` Wrapper Name Collision
- **Likelihood**: Medium — `graph.py` imports both `agreement_gate` (the wrapper from `a2a_client`) and `route_after_agreement` (the routing function from `agents.agreement_gate`).
- **Mitigation**: Import `route_after_agreement` explicitly: `from .agents.agreement_gate import route_after_agreement`. The `agreement_gate` function comes from `a2a_client`. No name collision because we don't import `agreement_gate` from `agents.agreement_gate`.

### Risk 5: `evaluate_agent` Wrapper Unused by Graph
- **Likelihood**: Certain — no graph node calls `evaluate_agent`. It's wrapped for API completeness (port 9002).
- **Mitigation**: Docstring note: `"""A2A wrapper for evaluate_agent (port 9002). Note: no graph node calls this — MCP server uses it in-process."""`. No special handling needed.

### Risk 6: `social_loop_agent` Name vs `social_search_agent` Name Confusion
- **Likelihood**: Low — `social_search_agent` is the M1 agent (local), `social_loop_agent` is the M2 agent (A2A).
- **Mitigation**: Import names are unambiguous: `social_search_agent` from `.agents`, `social_loop_agent` from `.a2a_client`.

---

## Validation Architecture

### Nyquist Validation (Build → Run → Assert)

**What to build**: `a2a_client.py` with all 10 wrappers + `graph.py` import changes + timeout config.

**How to validate**:

1. **Unit test — `AgentUnavailableError` propagation**: Mock `httpx.Client.post` to raise `ConnectError`; assert `search_agent(state)` returns degrade diff, not raises.

2. **Unit test — successful round-trip**: Mock `httpx.Client.post` to return a completed task with known artifact data; assert wrapper returns the expected diff dict.

3. **Unit test — EvidenceGraph rebuild**: Mock search_agent response with evidence list but no evidence_graph; assert wrapper adds `evidence_graph` as `EvidenceGraph` instance.

4. **Unit test — degrade decorator**: Call each wrapper with mocked failure; assert degrade diffs have correct keys and types.

5. **Integration test — `build_graph()` with mocked A2A**: Same as existing `test_graceful_degrade.py` but with patches on `a2a_client.*` instead of `graph.*`. Assert verdict produced.

6. **Integration test — `build_debate_graph()` with mocked A2A**: Same as existing `test_debate_pipeline_integration.py` but with patches on `a2a_client.*`. Assert verdict produced.

7. **Smoke test — real agents**: With all 10 agent servers running (`scripts/start_agents.sh`), run `python -m factcheck_agents.cli "Tin tức test"`. Assert verdict produced within 60s.

8. **Smoke test — all agents down**: With no agent servers, run the CLI. Assert graceful degrade (NEI verdict, no crash).

9. **Existing test suite**: Run all 83 existing unit tests. Assert 0 regressions.

### Success Criteria Mapping (from ROADMAP.md)

| Criterion | Validation |
|-----------|-----------|
| 1. `a2a_client.py` provides 10 typed wrappers; each constructs valid Task, calls HTTP, returns deserialized type; errors → `AgentUnavailableError` | Tests 1-4 |
| 2. `graph.py` node functions contain no direct imports of agent functions; every agent invocation goes through `a2a_client.*`; `build_graph()` and `build_debate_graph()` signatures unchanged | Code review + Tests 5-6 |
| 3. `AgentUnavailableError` triggers graceful-degrade (pipeline continues; affected field marked unavailable) | Tests 7-8 |
| 4. CLI with all servers up → verdict within 60s; CLI with all servers down → `unavailable` gracefully | Test 7-8 |
| 5. Existing 83 unit tests still pass unchanged | Test 9 |

---

## RESEARCH COMPLETE
