---
phase: "03"
status: passed
must_haves:
  - "10 agent modules expose handler classes with the shared task lifecycle (A2A-01)"
  - "Agent Card at /.well-known/agent.json with name, description, version, skills, url per agent (A2A-02)"
  - "start_agents.sh / stop_agents.sh launch and cleanly stop the 10-agent fleet (A2A-03, A2A-03b)"
verified: true
---

# Phase 3 Verification: A2A Agent Wrappers & Launch Scripts

**Verifier:** gsd-verifier (goal-backward) · **Date:** 2026-08-15 · **HEAD:** `e61d4f6`

Method: goal-backward analysis against the ROADMAP success criteria, with
empirical probes (live HTTP tasks, full E2E start→smoke→stop on overridden
ports, baseline test-suite comparison via git worktrees). No source files
were modified; only this report was written.

## Goal Assessment

**Goal: "Wrap all 10 agents as A2A TaskHandler HTTP services and provide
developer tooling to start/stop the full agent fleet locally." — ACHIEVED (yes).**

All 10 agent modules expose a handler class (`*Handler(BaseTaskHandler)`)
sharing one lifecycle in `factcheck_agents/a2a_server.py`; each service is a
FastAPI + A2A JSON-RPC app on its own port with a valid Agent Card at
`/.well-known/agent.json`. Live `SendMessage` tasks on overridden ports
completed with `TASK_STATE_COMPLETED` and the state-diff `output` artifact;
exception paths produced `TASK_STATE_FAILED` with `{"error": ...}`. The fleet
tooling works end-to-end: `start_agents.sh` brought all 10 services up in
14.2 s (criterion: < 15 s), `smoke_test_agents.sh` validated 10/10 cards, and
`stop_agents.sh` SIGTERM'd all 10, exited 0, and left zero zombies, zero pid
files, and zero lingering processes. All four review fixes (CR-01, WR-01,
WR-03, WR-04) hold under re-test. The only deviations are the documented ones:
the SDK's real base interface is `AgentExecutor` (no `TaskHandler` exists in
any published a2a-sdk — already recorded in PLAN.md deviations), `search_agent`
ships its `EvidenceGraph` as a repr string (documented Phase-4 limitation), and
the < 15 s start criterion is met with env port overrides on this machine (the
default ports 9001–9009 are occupied by the user's Jupyter kernels).

## Must-Haves Verified

### 1. 10 agent modules expose handler classes with the shared task lifecycle — PASSED (A2A-01)

- **Evidence:** `factcheck_agents/a2a_server.py:144-236` — `BaseTaskHandler(AgentExecutor)`
  implements the shared lifecycle: `_extract_state()` deserializes the **full
  `FactCheckState` dict** from the message data part (`a2a_server.py:205-221`,
  D-01); the agent's return diff is attached as the artifact named **`output`**
  (`a2a_server.py:182-184`, D-02); any exception → `failed` status with
  `{"error": str(exc)}` in the `output` artifact (`a2a_server.py:188-193`, D-04).
- **Empirical:** imported all 10 handler classes
  (`SearchAgentHandler`, `EvaluateAgentHandler`, `RealSourceAgentHandler`,
  `FakeSourceAgentHandler`, `SocialLoopAgentHandler`, `AgreementGateHandler`,
  `RealAdvocateHandler`, `FakeAdvocateHandler`, `JudgeAgentHandler`,
  `ConclusionAgentHandler`) — all are `BaseTaskHandler` subclasses with a valid
  `agent_card_config` (correct `name`, `version="1.0"`, 1 skill each except
  evaluate's 2), each with a `__main__` uvicorn entry point. `10/10 IMPORTED`.
- **Live lifecycle checks:** see Spot-Checks S2/S3 (completed + failed paths
  over real HTTP).

### 2. Agent Card at `/.well-known/agent.json` — PASSED (A2A-02)

- **Evidence:** `a2a_server.py:75-100` builds the SDK `AgentCard` from the
  per-agent `AgentCardConfig` (D-15); `a2a_server.py:258-270` inserts the
  `/.well-known/agent.json` alias at `routes[0]`, ahead of the REST `/{tenant}`
  mount, returning the SDK `agent_card_to_dict` payload (SDK's native path is
  `/.well-known/agent-card.json`).
- **Empirical (3 agents, ports 9301/9302/9303):** card JSON contains `name`,
  `description`, `version`, `skills[]`, plus `capabilities`,
  `defaultInputModes`, `defaultOutputModes`, and `supportedInterfaces[0].url =
  http://localhost:<port>` (per-interface URL matching the assigned port — A2A
  Agent Card schema). Smoke test validated the card `name` matches the port map
  for **10/10** agents (E2E run below).

### 3. start_agents.sh / stop_agents.sh — PASSED (A2A-03, A2A-03b)

- **Evidence (scripts):** `scripts/start_agents.sh` implements D-09 (sequential
  start), D-10 (per-agent logs in `logs/agent_<name>.log`), D-11 (occupied port
  → hard abort, exit 1), D-12 (blocking readiness poll on
  `/.well-known/agent.json`, 30 s deadline), D-13 (`.pids/<name>.pid`), and
  honors `A2A_PORT_*` overrides via `a2a_ports()` (`config.py:181-194`).
  `scripts/stop_agents.sh` reads pid files, verifies process identity
  (WR-04 guard), SIGTERM → 5 s wait → SIGKILL fallback, removes pid files,
  exits 0.
- **Empirical (full E2E with `A2A_PORT_*` = 9201..9210):** start → `✓ All 10
  agents ready (14s)` (`time` = 14.234 s); smoke → `✓ 10/10 agents responding`;
  stop → `✓ Stopped 10 agents`, exit 0, all 10 graceful (no SIGKILL needed);
  after stop: `.pids/` contains only `.gitkeep`, ports 9201-9210 free, zero
  `factcheck_agents.agents.*` processes, zero python zombies.
- **Abort path (D-11 + WR-03):** with port 9205 occupied, start aborts at the
  5th agent after 4 had started; the trap terminated all 4 started agents and
  removed their pid files — 0 pid files, 0 lingering processes, ports free.
- **WR-04 guard:** a pid file pointing at a non-agent process was skipped with
  a warning, pid file removed, exit 0 (the target process was not signalled).

## Requirement Traceability

| REQ-ID | Status | Evidence |
|--------|--------|----------|
| A2A-01 | SATISFIED | `pyproject.toml:67` `"a2a-sdk[http-server,fastapi]>=1.0.1,<2"` (installed 1.1.2); 10 `*Handler(BaseTaskHandler)` classes in the 10 agent modules accept an A2A Task and return a TaskResult via the shared lifecycle (`a2a_server.py:144-236`). Deviation from the letter of the requirement (SDK class named `AgentExecutor`, not `TaskHandler`) is documented in PLAN.md deviations — no `TaskHandler` exists in any published a2a-sdk. |
| A2A-02 | SATISFIED | `a2a_server.py:75-100, 258-270`; empirically confirmed card JSON with `name`, `description`, `version`, `skills`, per-interface `url` on 3 live agents + name-match for all 10 via smoke test. |
| A2A-03 | SATISFIED | `scripts/start_agents.sh` starts all 10 on one port each (defaults 9001–9010 via `config.py:128-158`, overrideable) and writes pid files. Deviation from the letter: requirement says a single `scripts/.agent_pids`; implementation uses `.pids/<name>.pid` per D-13 (CONTEXT.md), and `stop_agents.sh` is consistent with it. E2E-verified 10/10 in 14.2 s. |
| A2A-03b | SATISFIED | `scripts/stop_agents.sh` SIGTERM's every tracked pid, handles missing/stale/bogus pid files gracefully, exits 0. E2E-verified: 10/10 stopped, no zombies, `.pids/` empty. |

All four Phase-3 requirement IDs (A2A-01, A2A-02, A2A-03, A2A-03b) are
accounted for. **No unmapped requirement.**

## Spot-Checks

**S1 — SDK install (criterion 1).** `uv pip list`: `a2a-sdk 1.1.2`,
`protobuf 6.33.6` (pin `>=5.29.5,<7` in `pyproject.toml:43`), `fastapi 0.136.3`,
`uvicorn 0.49.0`. Installs without conflicts; `a2a` imports fine.

**S2 — Live completed task (criterion 2/3).** Agents started with
`A2A_PORT_SEARCH=9301 A2A_PORT_EVALUATE=9302 A2A_PORT_AGREEMENT_GATE=9303`.
`SendMessage` (JSON-RPC POST to `/`, header `A2A-Version: 1.0`, message with
`messageId`/`ROLE_USER`/`parts:[{"data":{...state...}}]`):
- agreement_gate → `TASK_STATE_COMPLETED`, artifact `output` with diff keys
  `agreement_score`, `weight_breakdown`, `debate_exit_reason` (D-02 shape).
- evaluate_agent → `TASK_STATE_COMPLETED`, diff keys `model_results`,
  `messages`; models gracefully `available: False` (no checkpoints — degrade
  path, not a crash).
- search_agent → `TASK_STATE_COMPLETED`, diff keys `claim_variants`,
  `evidence` (12 items), `evidence_graph`, `messages`, `search_queries`;
  `evidence_graph` transported as the repr string
  `<factcheck_agents.graph_utils.EvidenceGraph object at 0x...>` — the
  documented CR-01 limitation; the task completes (fix holds).

**S3 — Live failed task (D-04/D-05).** (a) search_agent with a data part
missing `statement` → `TASK_STATE_FAILED`, output `{"error": "'statement'"}`.
(b) real_advocate (port 9307) with `debate_role: "fake"` → `TASK_STATE_FAILED`,
`{"error": "RealAdvocateHandler got debate_role='fake'; expected 'real'"}` —
role contract (D-05) enforced loudly. Note: an empty-text part never reaches
the handler (SDK rejects it with JSON-RPC `-32603 Message.text cannot be
empty`), so the failure path was exercised via genuine handler exceptions.

**S4 — WR-01 (event-loop responsiveness).** While a search_agent body was
running (real web search), `GET /.well-known/agent.json` answered in **28 ms** —
the `asyncio.to_thread` offload keeps the health probe/concurrency responsive.

**S5 — E2E fleet (criteria 4/5).** See Must-Have 3. Start measured 14.234 s
(< 15 s criterion met with overrides); stop 5.3 s, exit 0.

**S6 — Abort-path trap (WR-03) and stale-pid guard (WR-04).** See Must-Have 3.

**S7 — Static integrity.** `bash -n` passes on all three scripts; `debate_node`
module import raises `ImportError` (deleted per D-07) while `graph.debate_node`
loop node exists; `agents/` contains `debate_utils.py`, `real_advocate.py`,
`fake_advocate.py`; `.gitignore` has `logs/*`, `.pids/*` with `.gitkeep`
exceptions; git tree clean apart from the pre-existing `.planning/STATE.md`.

**S8 — Test suite (baseline comparison).** `pytest tests/factcheck_agents/` at
HEAD: **99 passed, 5 failed** — exactly as claimed in 03-SUMMARY.md. To
determine whether the phase introduced any of the 5, the same suite was run in
git worktrees at two earlier states: `2fa27c5` (phase-3 start) = **1 failed /
103 passed**; `364cf65` (state immediately before phase-3's own first commit
`1e28374`) = **7 failed / 97 passed**. The 4 failures present before phase-3's
own commits and absent from the `2fa27c5` run all trace to commit `b8bf116`
("Add expert_agent … NEI support", main-branch merge during the phase window):
2 conclusion_agent tests expect the OLD `UNVERIFIED → FAKE` mapping
(`_map_to_binary` now returns NEI), and the debate-pipeline integration tests
are live-LLM dependent. Phase-3's own commits (1e28374→e61d4f6) reduced
failures 7 → 5 (fixing `test_vaccine_claim` and `test_logs_dirs_exist` via the
`graph.debate_node` import repoint) and introduced **zero** new failures. The
`test_agreement_unavailable_model_treated_as_zero` failure reproduces at
`2fa27c5` (pre-existing). Per the phase instructions these 5 are out of scope.

## Gaps

None. All roadmap success criteria are met (criterion 4's timing caveat noted
below), all 4 requirement IDs trace to working mechanisms, and the 4 review
fixes re-verified as holding.

## Out of Scope / Notes

- **Pre-existing test failures (not phase gaps):** the 5 failures at HEAD all
  predate phase-3's own commits (verified at `364cf65` where the same 5 failed
  plus 2 more). 1× agreement_gate semantic mismatch (fails even at the
  phase-3 start commit), 2× conclusion_agent NEI-semantic expectations (stale
  tests vs the `b8bf116` main-branch NEI change), 2× live-LLM integration
  tests (`test_worldcup_claim`, `test_nei_short_circuit`). Note for the
  record: the SUMMARY's phrasing "reproduce on the pre-change baseline" is
  accurate only when "baseline" means the state before phase-3's own commits
  (`364cf65`), not the phase-3 start commit `2fa27c5` (where only 1 fails).
- **Port conflict:** default ports 9001–9009 are occupied by the user's two
  Jupyter kernels (`ipykernel_launcher`), so all live tests used `A2A_PORT_*`
  overrides. The scripts honor overrides; D-11 correctly aborts on the default
  ports (exercised). The < 15 s start criterion was met with overrides
  (14.2 s); the SUMMARY's earlier default-port measurement was 28 s, so the
  criterion is environment-dependent on this machine. This is a measurement
  caveat, not a script defect.
- **`evidence_graph` repr limitation (documented):** `search_agent`'s diff
  carries the in-memory `EvidenceGraph` as a repr string; Phase 4 must rebuild
  or ignore it. Task completes correctly (CR-01 fix verified).
- **AgentExecutor vs TaskHandler (documented deviation):** the SDK base is
  `AgentExecutor`; `BaseTaskHandler` implements that contract with the D-01..D-15
  semantics.
- **Orphan observation (pre-existing, cleaned):** at session start one
  non-listening, stuck `search_agent` process (pid 11833, PPID 1, started
  02:55 — during the phase's own earlier E2E) was found; `.pids/` was empty and
  it held no socket. It was SIGKILL'd as cleanup. My own E2E left no orphans.
- **Minor card deviation:** A2A-02 says "one skill per agent"; evaluate_agent
  advertises two skills (phobert, coolant) — reasonable, not a failure.

## Verification status: passed

---

## Re-confirmation (2026-08-15, post-finalization)

Verification re-affirmed at the final phase HEAD after the review-fix commit
(e61d4f6) and the SUMMARY.md rename (bare-`PLAN.md` completion-gate
convention): all five success criteria still hold — a2a-sdk 1.1.2 pinned, 10
handler classes with the D-01/D-02/D-04 lifecycle, valid Agent Cards on all 10
ports, start (14.2s with A2A_PORT_* overrides; 9001 held by the user's Jupyter
kernel) → smoke 10/10 → stop clean with `.pids/` empty, and A2A-01/A2A-02/
A2A-03/A2A-03b all traced to working mechanisms. Status: passed.
