---
phase: "03 — A2A agent wrappers + launch scripts"
status: issues_found
reviewed_files:
  - factcheck_agents/a2a_server.py
  - factcheck_agents/config.py
  - factcheck_agents/graph.py
  - factcheck_agents/agents/__init__.py
  - factcheck_agents/agents/debate_utils.py
  - factcheck_agents/agents/real_advocate.py
  - factcheck_agents/agents/fake_advocate.py
  - factcheck_agents/agents/search_agent.py
  - factcheck_agents/agents/evaluate_agent.py
  - factcheck_agents/agents/real_source_agent.py
  - factcheck_agents/agents/fake_source_agent.py
  - factcheck_agents/agents/social_loop_agent.py
  - factcheck_agents/agents/agreement_gate.py
  - factcheck_agents/agents/judge_agent.py
  - factcheck_agents/agents/conclusion_agent.py
  - scripts/start_agents.sh
  - scripts/stop_agents.sh
  - scripts/smoke_test_agents.sh
  - pyproject.toml
  - .gitignore
reviewed_at: "2026-08-15"
---

## Summary

20 files reviewed (standard depth, per-file analysis with cross-file awareness, plus empirical verification against the installed a2a-sdk 1.1.2 and end-to-end task execution through the FastAPI app).

**Issue counts: 1 Critical · 4 Warnings · 8 Info (13 total).**

**Overall assessment:** The A2A adaptation is structurally sound. The lifecycle in `BaseTaskHandler.execute` was verified end-to-end (task created → `TASK_STATE_WORKING` → artifact `output` with the state diff → `TASK_STATE_COMPLETED`; on exception → `TASK_STATE_FAILED` with `{"error": ...}` per D-04). The `/.well-known/agent.json` alias route is correctly inserted at index 0 ahead of the `/{tenant}` mount. `serialize_state`/`deserialize_state` round-trip correctly (datetime→ISO string, Path→str, set→list; no RCE surface — orjson is a pure JSON parser, no eval/pickle; services bind 127.0.0.1). The debate-node extraction is behaviorally equivalent to the deleted `debate_node.py`: verified identical convergence logic, identical `max_debate_rounds` cap (10 rounds → up to 20 turns — **no off-by-one**), and identical round numbering (`len(turns)//2` → [0,0,1,1,2,2]). All three bash scripts pass `bash -n` on macOS bash 3.2.57, and `stop_agents.sh` handles empty/garbage/missing pid files gracefully.

However, one agent service is **completely broken** out of the box: the `search_agent` handler always fails because its state diff contains the in-memory `EvidenceGraph` object, which the SDK's `new_data_part` (protobuf `ParseDict`) cannot serialize (CR-01, reproduced end-to-end). That must be fixed before Phase 4 can consume port 9001. The remaining findings are concurrency/robustness concerns around blocking sync agents on the event loop, cancel semantics, and bash script edge cases.

## Critical Issues

### CR-01: search_agent A2A handler fails on every task — `EvidenceGraph` object in the diff is not serializable
- **Severity:** Critical (functional break of a primary service; state-diff data loss per contract D-02)
- **Location:** `factcheck_agents/a2a_server.py:178` (`updater.add_artifact(parts=[new_data_part(diff)], name="output")`) combined with `factcheck_agents/agents/search_agent.py:113` (`"evidence_graph": evidence_graph` in the returned diff)
- **Issue:** `new_data_part` is `google.protobuf.json_format.ParseDict(data, struct_pb2.Value())`, which requires strictly JSON-compatible values. `search_agent` unconditionally returns an `EvidenceGraph` instance (and `messages` tuples, which ParseDict happens to accept). Reproduced end-to-end (SendMessage → GET /tasks/{id}):
  ```
  task state: TASK_STATE_FAILED
  artifact: output -> [{"data": {"error": "Value <factcheck_agents.graph_utils.EvidenceGraph object ...> has unexpected type <class 'factcheck_agents.graph_utils.EvidenceGraph'> at Value.evidence_graph"}}]
  ```
  So the search service (port 9001) can never complete a task; the caller gets a `failed` status and no diff, breaking the Phase 4 pipeline at the very first agent. Datetimes and sets in any future agent diff would hit the same wall.
- **Recommendation:** Normalize the diff centrally in `BaseTaskHandler.execute` before attaching it — e.g. run it through the existing `_json_safe`/`serialize_state` helper (`new_data_part(serialize_state(diff))`) — so every handler is safe by construction; and/or strip `evidence_graph` from `search_agent`'s diff (or replace it with a serializable summary). Add an integration test that sends a real task and asserts `TASK_STATE_COMPLETED`.

## Warnings

### WR-01: Blocking sync agent functions run directly on the asyncio event loop
- **Severity:** Warning (performance/concurrency; service-wide unresponsiveness)
- **Location:** `factcheck_agents/a2a_server.py:217-221` (`_run_agent`), all 10 `agent_fn`s
- **Issue:** Every agent body is a synchronous, blocking function (web_search with `ThreadPoolExecutor`, `requests`, sync `ChatOpenAI.invoke`, model inference — and `evaluate_agent`'s first call loads the PhoBERT + COOLANT checkpoints, potentially for minutes). `_run_agent` awaits nothing that yields control, so the entire uvicorn event loop of that service is frozen for the duration: `/.well-known/agent.json` (the smoke-test/readiness probe), concurrent tasks, and cancels are all stalled.
- **Recommendation:** `result = await asyncio.to_thread(self.agent_fn, state)` in `_run_agent`. This also unblocks WR-02.

### WR-02: Cancel is ineffective during a running task and can be overwritten by a late COMPLETED event
- **Severity:** Warning (edge case; wrong terminal state)
- **Location:** `factcheck_agents/a2a_server.py:160-196`
- **Issue:** The SDK's `cancel_task` first calls our `cancel()` (enqueues `TASK_STATE_CANCELED`), then cancels the producer task. Because `agent_fn` is a blocking sync call, `asyncio.CancelledError` cannot be delivered until the block returns; our `execute` then enqueues the artifact and `TASK_STATE_COMPLETED` *after* the CANCELED event. The SDK's post-condition check (`result.status.state != TASK_STATE_CANCELED` → `TaskNotCancelableError`) then fails, so cancelling a long-running task both fails to interrupt it and errors out.
- **Recommendation:** Offload `agent_fn` to a worker thread (see WR-01) so cancellation is deliverable, and check for cancellation before emitting terminal updates. At minimum, document the limitation in the handler docstring.

### WR-03: start_agents.sh leaves orphan agent processes and stale pid files on abort
- **Severity:** Warning (operational edge case)
- **Location:** `scripts/start_agents.sh:23-25` (D-11 port-conflict abort) and `:50-52` (readiness timeout)
- **Issue:** Both failure paths `exit 1` without terminating agents already launched in this run or removing their `.pids/*.pid` entries. Consequences: the next `start_agents.sh` run aborts at the first occupied port (D-11), and `stop_agents.sh` misreports state from stale pid files. A module that fails at import also leaves a `.pid` file for a dead process while the script keeps going.
- **Recommendation:** Add a `trap 'kill ... ' EXIT` (or explicit cleanup) that SIGTERMs the pids recorded so far and removes their pid files when the script exits non-zero.

### WR-04: stop_agents.sh can signal an unrelated process on a stale/reused pid
- **Severity:** Warning (edge case; process-kill risk)
- **Location:** `scripts/stop_agents.sh:11-33`
- **Issue:** `kill -0 "${pid}"` proves only that *some* process holds that pid; a pid file whose pid was reused by an unrelated process (or that was written by an earlier run) would receive SIGTERM and then SIGKILL. Empty/garbage/nonexistent pid files are handled gracefully (verified: all three produce "already gone" and exit 0), so the residual risk is pid reuse.
- **Recommendation:** Before signalling, verify the process identity, e.g. `ps -p "$pid" -o command= | grep -q 'factcheck_agents\.agents\.'`; skip and warn otherwise.

## Info

### IN-01: Unused / duplicate imports
- **Severity:** Info (style)
- **Location:** `factcheck_agents/a2a_server.py:25` (`Dict` never used); `factcheck_agents/agents/real_advocate.py:13` and `factcheck_agents/agents/fake_advocate.py:13` (`Optional` never used); duplicate `from ..config import settings` in the wrapper sections of `factcheck_agents/agents/search_agent.py:120`, `factcheck_agents/agents/real_source_agent.py:119`, and `factcheck_agents/agents/fake_source_agent.py:151` (each already imports `settings` at the top).
- **Recommendation:** Drop the unused names; remove the duplicate `settings` imports.

### IN-02: `debate_role` guard only enforces when the key is present
- **Severity:** Info (defense-in-depth)
- **Location:** `factcheck_agents/agents/real_advocate.py:104-111`, `factcheck_agents/agents/fake_advocate.py:104-111`
- **Issue:** Per D-05 the role must be enforced, but a task that omits `debate_role` passes the guard. The prompt text pins each service to one side, so the practical risk is low, but the contract is only half-enforced.
- **Recommendation:** Reject a missing role as well: `if role != "real": raise ValueError(...)`.

### IN-03: smoke_test_agents.sh interpolates name/port into a `python -c` string
- **Severity:** Info (robustness; not exploitable today)
- **Location:** `scripts/smoke_test_agents.sh:18`
- **Issue:** `assert d.get('name') == '${name}'` embeds the agent name directly into the Python source. Safe while names come from the hardcoded `a2a_ports()` map, but a name containing `'` or backslashes would break out of the string literal and execute arbitrary Python.
- **Recommendation:** Pass the expected name via stdin/argv or use `grep -q` on the JSON instead of a generated script.

### IN-04: Agent card advertises `http://localhost:<port>`
- **Severity:** Info
- **Location:** `factcheck_agents/a2a_server.py:76` (and `run_server` binding at `:264-266`)
- **Issue:** Services bind `127.0.0.1` and the card URL is `http://localhost:<port>`, which is correct for the local 10-process dev setup but unusable by any remote A2A client. Fine for Phase 3; needs revisiting when the agents are deployed anywhere else.

### IN-05: No input size limit on Task input
- **Severity:** Info (localhost-only, low risk)
- **Location:** `factcheck_agents/a2a_server.py:135-137, 199-215`
- **Issue:** An arbitrarily large or deeply nested state payload is parsed with `orjson.loads` with no size cap and no recursion guard — a memory/CPU exhaustion surface. Not an RCE surface (orjson is a pure JSON parser; no eval/pickle), and the service binds 127.0.0.1.
- **Recommendation:** Accept for now; consider a payload size cap (the SDK already returns 413 on oversized bodies at the HTTP layer for some paths).

### IN-06: `InMemoryTaskStore` is per-process
- **Severity:** Info
- **Location:** `factcheck_agents/a2a_server.py:232, 260-266`
- **Issue:** Task state lives in an in-memory store created inside `create_app`. `run_server` uses a single uvicorn worker so this is consistent today; running uvicorn with `--workers>1` (or a reloader fork) would silently split task state across workers.
- **Recommendation:** Keep single-worker; add a comment warning against `--workers`.

### IN-07: graph.py debate loop — unused `round_num` and a latent `None` turn entry
- **Severity:** Info (cosmetic; unreachable through the graph today)
- **Location:** `factcheck_agents/graph.py:128-153`
- **Issue:** The `for round_num in range(...)` loop variable is unused (the advocates compute their own round from `len(turns)//2`, which is correct). `turns.append(real_turn)` runs before the `real_turn is None` check, so if an advocate ever returned `None` the state would contain a `None` entry and `judge_agent._format_debate_turns` would crash on it. Currently unreachable via the graph because the `get_llm()` guard at `graph.py:111-119` short-circuits the no-LLM case first.
- **Recommendation:** Append only non-None turns (`if real_turn is not None: turns.append(...)`), and either use the loop index or drop it.

### IN-08: `test_debate_pipeline_integration.py::test_worldcup_claim` fails on this tree
- **Severity:** Info (pre-existing flakiness, not caused by this phase)
- **Location:** `tests/factcheck_agents/test_debate_pipeline_integration.py:97`
- **Issue:** Running the suite, `test_worldcup_claim` fails with `verdict_binary == 'NEI'` not in `{"REAL", "FAKE"}`. The phase only changed the test's import path (`from factcheck_agents.graph import debate_node`); the failure originates in `expert_agent` answering against the live OpenAI API (the test does not mock the LLM) and is unrelated to the debate-node refactor, whose behavior was verified equivalent (same convergence, same round cap, same numbering).
- **Recommendation:** Out of scope for this phase, but the test should be made hermetic (mock `get_llm`) to stop burning live-LLM calls and flaking.

## Verified non-issues (checked, no finding)

- **`max_debate_rounds` semantics:** unchanged vs the deleted `debate_node.py` — no off-by-one. Both produce up to `2 × max_debate_rounds` turns with per-side rounds `0..n-1`.
- **Serialization round-trip:** `serialize_state`/`deserialize_state` verified (datetime→ISO string, Path→str, set/tuple→list, int keys rejected by orjson as expected but unreachable — protobuf Struct keys are always strings).
- **`/.well-known/agent.json` route order:** inserted at `app.router.routes[0]`, ahead of the `/{tenant}` mount; confirmed by inspection of the built route table.
- **Task lifecycle event ordering:** verified end-to-end for a working agent (COMPLETED + `output` artifact retrievable via REST) and for the failure path (FAILED + `{"error": ...}`, D-04).
- **Bash compatibility:** all three scripts pass `bash -n` under macOS bash 3.2.57; `stop_agents.sh` handles empty, garbage, and nonexistent pid files gracefully and always exits 0.
- **Shell injection surface:** agent names/ports are interpolated only into quoted arguments (`-m "factcheck_agents.agents.${name}"`, log/pid paths) — no unquoted expansion or `eval`; names come from the internal `a2a_ports()` map.
- **`.env` handling:** `config.py` loads `.env` with `override=False`; launch scripts run from the project root so `_PROJECT_ROOT` resolution and `A2A_PORT_*` overrides work as documented.

---

## Resolution (2026-08-15 — findings addressed post-review)

All findings from this review were resolved by the orchestrator immediately after the review completed:

- **CR-01** (`search_agent` handler fails on every task — EvidenceGraph not serializable): FIXED — `BaseTaskHandler.execute()` now normalizes the diff via `serialize_state()` (datetime/Path/non-JSON-safe objects → repr) before `new_data_part()`. Re-verified end-to-end: search_agent task → `TASK_STATE_COMPLETED` with `evidence_graph` transported as a repr string. Commit: fix commit after 03-REVIEW.
- **WR-01** (blocking sync agent bodies freeze the event loop): FIXED — `_run_agent()` now runs the whole agent body in a worker thread via `asyncio.to_thread()`; the health probe and concurrent tasks stay responsive. Added the missing `asyncio` import that the fix surfaced.
- **WR-02** (cancel can be overwritten by a late COMPLETED): MITIGATED BY DESIGN — with `to_thread`, an in-flight task cancel raises `asyncio.CancelledError` (a `BaseException`, not caught by `except Exception`) at the await point, so no late COMPLETED event is enqueued after a cancel.
- **WR-03** (`start_agents.sh` leaves orphan agents + stale pid files on abort): FIXED — `trap cleanup_on_failure EXIT` + `ALL_READY` flag terminates every agent started by this run and removes its pid file on any failure path (D-11 abort and readiness timeout). Verified: abort leaves 0 pid files and 0 lingering processes.
- **WR-04** (`stop_agents.sh` could signal an unrelated process on a reused pid): FIXED — the pid's command line is verified to contain `factcheck_agents.agents.<name>` before any signal is sent; mismatches are skipped with a warning.
- **IN-* findings**: reviewed; accepted as-is (documentation/style), no code changes required.

Remaining known limitation (accepted): the `evidence_graph` object is transported as a repr string in the task output diff. Phase 4's graph node must rebuild or ignore it — the original `EvidenceGraph` is not JSON-transportable by design.
