# Phase 3: A2A Agent Wrappers & Launch Scripts - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-15
**Phase:** 3-a2a-agent-wrappers-launch-scripts
**Areas discussed:** Task input contract, debate_node split, Launch script behavior, Base server module scope

---

## Task Input Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Full FactCheckState | Serialize entire state dict as JSON. Simple, zero new types. | ✓ (Claude discretion) |
| Per-agent minimal fields | Each handler declares only needed fields. Cleaner API, but 10 schemas to maintain. | |
| You decide | Claude picks based on Phase 4 migration cost. | |

**User's choice:** You decide — Claude chose Full FactCheckState for simplest migration path.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Return state diff only | TaskResult.output = only the keys the agent mutated. | ✓ |
| Return full updated state | TaskResult.output = full FactCheckState after agent ran. | |

**User's choice:** Return state diff only.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Each agent file | Handler handles its own JSON parsing inline. | |
| Shared base module | a2a_server.py exports serialize/deserialize helpers. | ✓ (Claude discretion) |

**User's choice:** You decide — Claude chose shared base module for single update point.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Error status + message | TaskResult.status = 'failed'; output = {'error': str(e)}. | ✓ |
| Partial state with unavailable sentinel | TaskResult.status = 'completed'; output with 'unavailable' values. | |

**User's choice:** Error status + message.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Encoded in Task.input | debate_role: 'real'/'fake' key in input dict. | ✓ |
| Separate handlers per role | Two distinct TaskHandler classes, each hardcoded. | |

**User's choice:** Encoded in Task.input — handler reads role at runtime.

---

| Option | Description | Selected |
|--------|-------------|----------|
| In Task.input each call | Full history sent every time. Stateless, easy to debug. | ✓ |
| Server-side state | Handler stores history internally across calls. | |

**User's choice:** In Task.input each call. Payload capped at max_debate_rounds=2.

---

## debate_node Split

| Option | Description | Selected |
|--------|-------------|----------|
| Split into two files | Create real_advocate.py and fake_advocate.py. | ✓ |
| Single file, two handlers | One file exports two TaskHandler classes. | |
| Single file, role param | One handler, role from Task.input.debate_role. | |

**User's choice:** Split into two files.

---

| Option | Description | Selected |
|--------|-------------|----------|
| New debate_utils.py | Extract shared code into separate utility file. | ✓ |
| Keep in debate_node.py | Both files import from debate_node.py. | |
| Inline in each file | Copy shared logic. ~30 extra lines of duplication. | |

**User's choice:** New debate_utils.py.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Delete it | Remove debate_node.py entirely. Clean break. | ✓ |
| Keep as compat shim | debate_node.py becomes a re-export module. | |

**User's choice:** Delete it.

---

## Launch Script Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential | Start one at a time. Easy to debug. | ✓ |
| Parallel (bg jobs) | Start all 10 in background. Faster. | |

**User's choice:** Sequential.

---

| Option | Description | Selected |
|--------|-------------|----------|
| stdout/stderr directly | Each process writes to terminal. | |
| Per-agent log files | logs/agent_<name>.log per agent. | ✓ (Claude discretion) |
| Suppress by default | Quiet unless --verbose flag. | |

**User's choice:** You decide — Claude chose per-agent log files.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Fail immediately | Print error, stop whole script. | ✓ |
| Skip and warn | Print warning, continue with remaining agents. | |

**User's choice:** Fail immediately.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Block until all ready | Poll /.well-known/agent.json until HTTP 200 or timeout. | ✓ |
| Fire and forget | Start all agents and exit immediately. | |

**User's choice:** Block until all ready.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Single PID file | One .agent_pids file listing all 10 PIDs. | |
| Per-agent PID files | .pids/search_agent.pid, .pids/evaluate_agent.pid, etc. | ✓ |

**User's choice:** Per-agent PID files.

---

## Base Server Module Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal — uvicorn + cards | Just app factory and Agent Card builder. | |
| Full — handler base class | Minimal + BaseTaskHandler with shared process() flow. | ✓ (Claude discretion) |
| You decide | Claude picks the right balance. | |

**User's choice:** You decide — Claude chose full BaseTaskHandler with shared process() flow.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Each agent defines own | Each handler file defines own card dict inline. | |
| Cards in a2a_server.py | All 10 card dicts live in a2a_server.py as AGENT_CARDS. | |
| Per-agent config dataclass | Shared AgentCardConfig dataclass, per-agent instances. | ✓ (Claude discretion) |

**User's choice:** You decide — Claude chose per-agent AgentCardConfig dataclass.

---

## Claude's Discretion

- Task.input uses full FactCheckState (not per-agent minimal fields)
- Serialization helpers in shared a2a_server.py
- Per-agent log files for uvicorn output
- Full BaseTaskHandler abstract class (not just uvicorn + cards)
- AgentCardConfig dataclass for Agent Card content

## Deferred Ideas

None — discussion stayed within phase scope.
