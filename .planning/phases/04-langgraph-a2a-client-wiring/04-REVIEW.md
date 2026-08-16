---
phase: '04-langgraph-a2a-client-wiring'
status: reviewed
depth: standard
reviewed: 2026-08-17
files:
  - factcheck_agents/a2a_client.py
  - factcheck_agents/config.py
  - factcheck_agents/graph.py
  - factcheck_agents/graph_utils.py
  - factcheck_agents/cli.py
  - factcheck_agents/agents/conclusion_agent.py
---

# Phase 04 Code Review

Review of the source files changed by Phase 04 (A2A client wiring). Performed
inline (standard depth) — reviewer agent unavailable in this runtime; the gate
is advisory. All findings below are observations, not blockers; behavior was
additionally validated by the live CLI smoke test and the full test suite.

## Findings

### [Info] Shallow-copied degrade diffs share nested dicts across calls
- **File:** `factcheck_agents/a2a_client.py` (`degrade_on_unavailable`)
- `return dict(degrade_diff)` protects the top-level keys but the nested
  `verdict` / `weight_breakdown` / `debate_turn` dicts are shared module
  constants. No current consumer mutates them in place (verified: graph nodes
  read-only; conclusion_agent builds its own verdict server-side), but a
  future in-place write would corrupt the constant for all later calls.
- **Suggestion:** `copy.deepcopy(degrade_diff)` or freeze the constants. Low
  priority — matches the plan's D-04 spec (shallow copy).

### [Info] `httpx.Client` instances are never closed
- **File:** `factcheck_agents/a2a_client.py` (`call_agent`)
- A new `httpx.Client(timeout=...)` is created per call without a context
  manager, so connections are released by GC rather than explicitly. Chosen
  deliberately (research Q3: stateless per-call client, YAGNI). With ~15 calls
  per pipeline run the leak is negligible; if profiling ever shows FD growth,
  wrap in `with httpx.Client(...) as client:`.

### [Info] `_timeout_for` raises AttributeError for unmapped agent names
- **File:** `factcheck_agents/a2a_client.py`
- `_SUFFIX_BY_AGENT.get(agent_name, agent_name)` falls back to the raw name for
  unknown agents, so `getattr(settings, "a2a_client_timeout_<name>")` raises
  AttributeError instead of a friendly error. All 10 current agents are mapped;
  acceptable, but a KeyError-style message would be clearer for future agents.

### [Info] cli.py thread_id derived from `hash(statement)`
- **File:** `factcheck_agents/cli.py`
- `hash()` on str is process-randomized (PYTHONHASHSEED), so thread ids differ
  across runs — this is exactly what the CLI wants (no cross-run checkpoint
  collisions). If deterministic per-claim checkpoints are ever desired, switch
  to a stable hash (e.g., hashlib.md5 of the statement).

### [Info] EvidenceGraph checkpoint type is unregistered for langgraph strict mode
- **File:** `factcheck_agents/graph_utils.py`
- The jsonplus serializer emits a warning that unregistered types will be
  blocked in a future langgraph version (`LANGGRAPH_STRICT_MSGPACK=true` blocks
  now). If strict mode is ever enabled, the module must be added to
  `allowed_msgpack_modules` or the graph serialized as plain data. Tracked in
  `deferred-items.md`.

### [Info] `TASK_STATE_WORKING` returns empty diff without polling
- **File:** `factcheck_agents/a2a_client.py`
- Per plan (Risk 2 mitigation), a non-terminal task returns `{}` — the caller
  sees a silently empty diff rather than a retry. Servers block until
  completion (confirmed live), so this path is theoretical. If the SDK ever
  returns early, results will be silently empty; consider polling `GET
  /tasks/{id}` or raising instead. Documented trade-off from the plan.

## No Critical / Warning findings

- Protocol compliance (`A2A-Version`, `ROLE_USER`, `{"task": ...}` unwrap)
  verified against a live SDK 1.1.2 server.
- `debate_node` partial-debate edge cases (both-down, one-sided, convergence
  skip, IN-07 non-None appends) exercised with mocked advocates.
- Checkpoint round-trip (`_asdict`/`graph_data`) round-tripped through the real
  `JsonPlusSerializer`.
- CLI up/down smoke tests passed; full suite at exact baseline parity.
